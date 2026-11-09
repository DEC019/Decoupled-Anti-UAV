import argparse
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO


IMAGE_SUFFIXES = {'.bmp', '.dng', '.jpeg', '.jpg', '.mpo', '.png', '.tif', '.tiff', '.webp'}


def parse_class_list(value):
	if value is None or value == '':
		return set()
	return {int(item.strip()) for item in value.split(',') if item.strip()}


def parse_class_sequence(value):
	if value is None or value == '':
		return []
	return [int(item.strip()) for item in str(value).split(',') if item.strip()]


def parse_ratio_list(value):
	if isinstance(value, (int, float)):
		return [float(value)]
	if value is None or value == '':
		return [0.0]
	ratio_list = [float(item.strip()) for item in str(value).split(',') if item.strip()]
	if not ratio_list:
		return [0.0]
	if any(ratio <= -0.5 for ratio in ratio_list):
		raise ValueError('All expand ratios must be greater than -0.5')
	return ratio_list


def list_source_images(source):
	source_path = Path(source)
	if source_path.is_file():
		return [source_path]
	if source_path.is_dir():
		return sorted(path for path in source_path.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
	raise FileNotFoundError(f'Source path does not exist: {source}')


def expand_box(x, y, w, h, img_w, img_h, ratio=0.0):
	cx = x + w / 2
	cy = y + h / 2

	new_w = w * (1 + ratio * 2)
	new_h = h * (1 + ratio * 2)

	x1 = max(0, int(cx - new_w / 2))
	y1 = max(0, int(cy - new_h / 2))
	x2 = min(img_w, int(cx + new_w / 2))
	y2 = min(img_h, int(cy + new_h / 2))
	return x1, y1, x2, y2


def resize_and_pad_cv(image, target_size=224):
	height, width = image.shape[:2]
	if height == 0 or width == 0:
		raise ValueError('Cannot resize an empty crop')

	if width >= height:
		scale = target_size / width
		new_width = target_size
		new_height = max(1, int(height * scale))
	else:
		scale = target_size / height
		new_height = target_size
		new_width = max(1, int(width * scale))

	resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LANCZOS4)
	canvas = cv2.copyMakeBorder(
		resized,
		(target_size - new_height) // 2,
		target_size - new_height - (target_size - new_height) // 2,
		(target_size - new_width) // 2,
		target_size - new_width - (target_size - new_width) // 2,
		cv2.BORDER_CONSTANT,
		value=(0, 0, 0),
	)
	return canvas


def compute_iou(box_a, box_b):
	ax1, ay1 = box_a['x'], box_a['y']
	ax2, ay2 = box_a['x'] + box_a['w'], box_a['y'] + box_a['h']
	bx1, by1 = box_b['x'], box_b['y']
	bx2, by2 = box_b['x'] + box_b['w'], box_b['y'] + box_b['h']

	inter_x1 = max(ax1, bx1)
	inter_y1 = max(ay1, by1)
	inter_x2 = min(ax2, bx2)
	inter_y2 = min(ay2, by2)

	inter_w = max(0, inter_x2 - inter_x1)
	inter_h = max(0, inter_y2 - inter_y1)
	inter_area = inter_w * inter_h
	if inter_area == 0:
		return 0.0

	area_a = box_a['w'] * box_a['h']
	area_b = box_b['w'] * box_b['h']
	union = area_a + area_b - inter_area
	if union <= 0:
		return 0.0
	return inter_area / union


def apply_classwise_nms(items, iou_threshold):
	if iou_threshold == -1:
		return sorted(items, key=lambda item: (item['cls_id'], -item['cls_conf']))

	kept = []
	for cls_id in sorted({item['cls_id'] for item in items}):
		class_items = [item for item in items if item['cls_id'] == cls_id]
		class_items.sort(key=lambda item: item['cls_conf'], reverse=True)
		selected = []
		for item in class_items:
			if all(compute_iou(item, chosen) <= iou_threshold for chosen in selected):
				selected.append(item)
		kept.extend(selected)
	kept.sort(key=lambda item: (item['cls_id'], -item['cls_conf']))
	return kept


def draw_detection_candidates(image_bgr, candidate_boxes, det_model):
	vis = image_bgr.copy()
	for candidate in candidate_boxes:
		x = candidate['x']
		y = candidate['y']
		w = candidate['w']
		h = candidate['h']
		det_cls = candidate['det_cls']
		det_conf = candidate['det_conf']
		cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 255), 2)
		class_name = det_model.names.get(det_cls, str(det_cls)) if isinstance(det_model.names, dict) else str(det_cls)
		label = f'{class_name}:{det_conf:.2f}'
		cv2.putText(vis, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
	return vis


def format_candidate_line(candidate):
	return (
		f"det_cls={candidate['det_cls']} det_conf={candidate['det_conf']:.6f} "
		f"xywh=({candidate['x']},{candidate['y']},{candidate['w']},{candidate['h']})"
	)


def filter_detector_candidates(candidate_boxes, reject_classes, reject_conf):
	if not reject_classes:
		return candidate_boxes, []

	kept = []
	rejected = []
	for candidate in candidate_boxes:
		if candidate['det_cls'] in reject_classes and candidate['det_conf'] >= reject_conf:
			rejected.append(candidate)
		else:
			kept.append(candidate)
	return kept, rejected


def save_debug_outputs(output_debug_dir, output_stem, image_bgr, det_vis, cls_vis, candidate_boxes, rejected_boxes, filtered_boxes, kept):
	output_debug_dir.mkdir(parents=True, exist_ok=True)
	cv2.imwrite(str(output_debug_dir / f'{output_stem}_image.png'), image_bgr)
	cv2.imwrite(str(output_debug_dir / f'{output_stem}_det_vis.png'), det_vis)
	cv2.imwrite(str(output_debug_dir / f'{output_stem}_cls_vis.png'), cls_vis)

	debug_lines = [
		f'candidates={len(candidate_boxes)}',
		f'rejected_by_detector_filter={len(rejected_boxes)}',
		f'sent_to_classifier={len(filtered_boxes)}',
		f'final_kept={len(kept)}',
		'',
		'[det_candidates]',
	]
	for candidate in candidate_boxes:
		debug_lines.append(format_candidate_line(candidate))
	debug_lines.append('')
	debug_lines.append('[rejected_before_cls]')
	for candidate in rejected_boxes:
		debug_lines.append(format_candidate_line(candidate))
	debug_lines.append('')
	debug_lines.append('[sent_to_classifier]')
	for candidate in filtered_boxes:
		debug_lines.append(format_candidate_line(candidate))
	debug_lines.append('')
	debug_lines.append('[final_after_cls_nms]')
	for item in kept:
		debug_lines.append(
			f"cls_id={item['cls_id']} cls_conf={item['cls_conf']:.6f} det_conf={item['det_conf']:.6f} "
			f"expand_ratio={item['expand_ratio']:.4f} xywh=({item['x']},{item['y']},{item['w']},{item['h']})"
		)
	(output_debug_dir / f'{output_stem}_debug.txt').write_text('\n'.join(debug_lines) + '\n', encoding='utf-8')

	if kept:
		mosaic_images = []
		for item in kept:
			x1 = item['x']
			y1 = item['y']
			x2 = item['x'] + item['w']
			y2 = item['y'] + item['h']
			crop = image_bgr[y1:y2, x1:x2]
			if crop.size == 0:
				continue
			panel = resize_and_pad_cv(crop, target_size=224)
			caption = f"c{item['cls_id']}:{item['cls_conf']:.2f}"
			cv2.putText(panel, caption, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
			mosaic_images.append(panel)
		if mosaic_images:
			mosaic = np.concatenate(mosaic_images, axis=1)
			cv2.imwrite(str(output_debug_dir / f'{output_stem}_kept_crops.png'), mosaic)


def collect_candidate_boxes(det_result):
	if det_result.boxes is None or det_result.boxes.xyxy is None:
		return []

	xyxy_list = det_result.boxes.xyxy.tolist()
	conf_list = det_result.boxes.conf.tolist() if det_result.boxes.conf is not None else [0.0] * len(xyxy_list)
	cls_list = det_result.boxes.cls.tolist() if det_result.boxes.cls is not None else [-1] * len(xyxy_list)

	boxes = []
	for xyxy, det_conf, det_cls in zip(xyxy_list, conf_list, cls_list):
		x1, y1, x2, y2 = xyxy
		x = int(round(x1))
		y = int(round(y1))
		w = max(1, int(round(x2 - x1)))
		h = max(1, int(round(y2 - y1)))
		boxes.append({
			'x': x,
			'y': y,
			'w': w,
			'h': h,
			'area': w * h,
			'det_conf': float(det_conf),
			'det_cls': int(det_cls),
		})
	return boxes


def classify_candidates(args, cls_model, image_bgr, candidate_boxes):
	crops = []
	crop_infos = []
	image_h, image_w = image_bgr.shape[:2]
	expand_ratios = parse_ratio_list(args.expand_ratio)
	target_classes = parse_class_list(args.target_class)

	for candidate in candidate_boxes:
		for ratio in expand_ratios:
			x1, y1, x2, y2 = expand_box(candidate['x'], candidate['y'], candidate['w'], candidate['h'], image_w, image_h, ratio)
			crop = image_bgr[y1:y2, x1:x2]
			if crop.size == 0:
				continue
			crops.append(resize_and_pad_cv(crop, target_size=args.cls_size))
			crop_infos.append((candidate, ratio))

	vis = image_bgr.copy()
	label_lines = []
	kept = []

	if not crops:
		return vis, label_lines, kept

	best_by_box = {}
	cls_batch_limit = getattr(args, 'cls_batch', 0)
	cls_batch = len(crops) if cls_batch_limit <= 0 else min(cls_batch_limit, len(crops))
	results_list = []
	for batch_start in range(0, len(crops), cls_batch):
		batch_crops = crops[batch_start:batch_start + cls_batch]
		results_list.extend(cls_model(
			batch_crops,
			imgsz=args.cls_size,
			device=args.device,
			batch=len(batch_crops),
			verbose=False,
		))
	for (candidate, ratio), cls_result in zip(crop_infos, results_list):
		if cls_result.probs is None:
			continue

		cls_id = int(cls_result.probs.top1)
		cls_conf = float(cls_result.probs.top1conf)
		if cls_conf < args.cls_conf:
			continue
		if target_classes and cls_id not in target_classes:
			continue

		box_key = (candidate['x'], candidate['y'], candidate['w'], candidate['h'])
		current_best = best_by_box.get(box_key)
		if current_best is None or cls_conf > current_best['cls_conf']:
			best_by_box[box_key] = {
				**candidate,
				'cls_id': cls_id,
				'cls_conf': cls_conf,
				'expand_ratio': ratio,
			}

	kept = apply_classwise_nms(list(best_by_box.values()), args.nms_iou)

	for item in kept:
		x = item['x']
		y = item['y']
		w = item['w']
		h = item['h']
		cls_id = item['cls_id']
		cls_conf = item['cls_conf']

		label_lines.append(
			f"{cls_id} {(x + w / 2) / image_w:.8f} {(y + h / 2) / image_h:.8f} "
			f"{w / image_w:.8f} {h / image_h:.8f} {cls_conf:.8f}"
		)

		cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 0, 255), 2)
		class_name = cls_model.names.get(cls_id, str(cls_id)) if isinstance(cls_model.names, dict) else str(cls_id)
		label = f'{class_name}:{cls_conf:.2f}|det:{item["det_conf"]:.2f}'
		cv2.putText(vis, label, (x, max(0, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

	return vis, label_lines, kept


def run(args):
	image_paths = list_source_images(args.source)
	if not image_paths:
		raise ValueError(f'No images found in source: {args.source}')

	output_dir = Path(args.output)
	output_txt_dir = output_dir / 'txt'
	output_vis_dir = output_dir / 'vis'
	output_debug_dir = output_dir / 'debug'
	output_txt_dir.mkdir(parents=True, exist_ok=True)
	output_vis_dir.mkdir(parents=True, exist_ok=True)

	det_model = YOLO(args.det_model)
	cls_model = YOLO(args.cls_model)
	reject_classes = set(parse_class_sequence(args.reject_det_classes))

	summary_lines = []
	for image_path in image_paths:
		result = det_model.predict(
			source=str(image_path),
			conf=args.det_conf,
			imgsz=args.det_size,
			device=args.device,
			verbose=False,
			max_det=args.max_det,
			end2end=args.det_end2end,
		)[0]

		image_bgr = result.orig_img
		candidate_boxes = collect_candidate_boxes(result)
		filtered_boxes, rejected_boxes = filter_detector_candidates(candidate_boxes, reject_classes, args.reject_det_conf)
		det_vis = draw_detection_candidates(image_bgr, filtered_boxes, det_model)
		vis, label_lines, kept = classify_candidates(args, cls_model, image_bgr, filtered_boxes)

		output_stem = image_path.stem
		if args.debug:
			save_debug_outputs(output_debug_dir, output_stem, image_bgr, det_vis, vis, candidate_boxes, rejected_boxes, filtered_boxes, kept)
		else:
			(output_txt_dir / f'{output_stem}.txt').write_text('\n'.join(label_lines), encoding='utf-8')
			cv2.imwrite(str(output_vis_dir / f'{output_stem}_vis.png'), vis)

		summary_line = (
			f'{image_path.name}: candidates={len(candidate_boxes)}, filtered={len(rejected_boxes)}, '
			f'classified={len(kept)}, det_conf={args.det_conf:.4f}, cls_conf={args.cls_conf:.4f}, nms_iou={args.nms_iou:.2f}'
		)
		summary_lines.append(summary_line)
		print(summary_line)

	(output_dir / 'summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
	print(f'Saved results to: {output_dir}')


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--source', type=str, default='./data/')
	parser.add_argument('--output', type=str, default='./result/')
	parser.add_argument('--device', type=str, default='cuda:6')
	parser.add_argument('--det-model', type=str, default='./weights/detection/best.pt')
	parser.add_argument('--cls-model', type=str, default='./weights/classify/best.pt')
	parser.add_argument('--det-end2end', type=bool, default=False)
	parser.add_argument('--det-size', type=int, default=640)
	parser.add_argument('--cls-size', type=int, default=224)
	parser.add_argument('--cls-batch', type=int, default=0)
	parser.add_argument('--det-conf', type=float, default=0.005)
	parser.add_argument('--max-det', type=int, default=300)
	parser.add_argument('--reject-det-classes', type=str, default='0,1,2,3,5,6,7,8,9,10,11,12,13,24,25,28,56,57,59,60')
	parser.add_argument('--reject-det-conf', type=float, default=0.3)
	parser.add_argument('--nms-iou', type=float, default=0.1)
	parser.add_argument('--cls-conf', type=float, default=0.0)
	parser.add_argument('--target-class', type=str, default='2')
	parser.add_argument('--expand-ratio', type=str, default='-0.15, 0, 0.15')
	parser.add_argument('--debug', action='store_true')
	args = parser.parse_args()
	run(args)


if __name__ == '__main__':
	main()
