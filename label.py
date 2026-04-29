# nix-shell -p python313 python313Packages.numpy python313Packages.matplotlib python313Packages.opencv4

import json
import os
import glob
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

plt.rcParams['keymap.save'] = []

widget_w, widget_h = 198, 102

# ----------------- 模式選擇 -----------------
print("選擇模式：")
print("  1  label  （手動標註）")
print("  2  recalc （重新計算藍色 poly）")
while True:
    mode_input = input("模式 (1/2)：").strip()
    if mode_input in ("1", "2", "label", "recalc"):
        break
    print("請輸入 1 或 2")
MODE = "label" if mode_input in ("1", "label") else "recalc"
print(f">> 模式：{MODE}")

start = int(input("從哪張圖："))
end = int(input("到哪張圖："))

files = []
for i in range(start, end + 1):
    folder = f"./{i:03d}"
    for pattern in ["H.avif", "V.avif"]:
        path = os.path.join(folder, pattern)
        if os.path.exists(path):
            files.append(path)

print(f"共找到 {len(files)} 張圖")

drawing_poly = []
polygon_groups = []
current_action = None


def read_image(path):
    """用 Pillow 讀取 AVIF（及其他格式），回傳 RGB numpy array"""
    try:
        with Image.open(path) as im:
            im = im.convert("RGB")
            return np.array(im)
    except Exception as e:
        print(f"無法讀取：{path}，原因：{e}")
        return None


def get_json_paths(image_file):
    """取得紅色與藍色 JSON 路徑"""
    base = os.path.splitext(image_file)[0]
    red_json = base + ".json"
    # _H.json / _V.json / _O.json
    name = os.path.basename(base)          # e.g. H
    dir_ = os.path.dirname(base)
    blue_json = os.path.join(dir_, f"_{name}.json")
    return red_json, blue_json


def load_existing_json(image_file):
    red_json, blue_json = get_json_paths(image_file)
    red_polys_loaded = []
    blue_polys_loaded = []

    if os.path.exists(red_json):
        try:
            with open(red_json, "r", encoding="utf-8") as f:
                red_polys_loaded = json.load(f).get("polys", [])
            print(f"載入 {len(red_polys_loaded)} 個紅色多邊形從 {red_json}")
        except Exception as e:
            print(f"載入紅色多邊形失敗: {e}")

    if os.path.exists(blue_json):
        try:
            with open(blue_json, "r", encoding="utf-8") as f:
                blue_polys_loaded = json.load(f).get("polys", [])
            print(f"載入 {len(blue_polys_loaded)} 個藍色多邊形從 {blue_json}")
        except Exception as e:
            print(f"載入藍色多邊形失敗: {e}")

    return red_polys_loaded, blue_polys_loaded


def resize_image(img):
    h, w = img.shape[:2]
    return cv2.resize(img, (1920, 1080)) if w >= h else cv2.resize(img, (1080, 1920))

def remove_collinear_points(pts, tol=1e-6):
    """移除多邊形中的共線點（cross product 接近 0）"""
    if len(pts) < 3:
        return pts
    result = []
    n = len(pts)
    for i in range(n):
        a = np.array(pts[(i - 1) % n])
        b = np.array(pts[i])
        c = np.array(pts[(i + 1) % n])
        cross = abs((b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0]))
        if cross > tol:
            result.append(pts[i])
    return result


def generate_blue_poly(red_poly, img_shape, widget_w, widget_h, tone="dark"):
    h_img, w_img = img_shape[:2]
    red_poly_np = np.array(red_poly, dtype=np.int32)
    mask = np.zeros((h_img, w_img), dtype=np.uint8)
    cv2.fillPoly(mask, [red_poly_np], 255)

    pad_x = widget_w // 2
    pad_y = widget_h // 2
    padded_mask = np.zeros((h_img + 2*pad_y, w_img + 2*pad_x), dtype=np.uint8)
    padded_mask[pad_y:pad_y+h_img, pad_x:pad_x+w_img] = mask

    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (widget_w, widget_h))
    mask_eroded = cv2.erode(padded_mask, kernel)
    mask_eroded = mask_eroded[pad_y:pad_y+h_img, pad_x:pad_x+w_img]

    contours, _ = cv2.findContours(mask_eroded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    blue_polys = []
    for cnt in contours:
        pts = cnt.reshape(-1, 2).tolist()

        # 必須至少 3 點才算有效多邊形，2 點線段直接忽略
        if len(pts) >= 3:
            approx = cv2.approxPolyDP(cnt, 1, True)
            approx_pts = approx.reshape(-1, 2).tolist()
            approx_pts = remove_collinear_points(approx_pts)
            if len(approx_pts) >= 3:
                blue_polys.append({"points": approx_pts, "tone": tone})

    return blue_polys


def compute_blue_ratios(all_blue_polys):
    areas = []
    for poly in all_blue_polys:
        pts = np.array(poly["points"], dtype=np.int32)
        area = cv2.contourArea(pts) if len(pts) >= 3 else 0
        areas.append(area)

    total_area = sum(areas)
    for i, poly in enumerate(all_blue_polys):
        poly["ratio"] = areas[i] / total_area if total_area > 0 else 0
    return all_blue_polys


def create_polygon_groups_from_loaded(red_polys, blue_polys, img_shape):
    groups = []
    for red_poly in red_polys:
        tone = red_poly.get("tone", "dark")
        blue = generate_blue_poly(red_poly["points"], img_shape, widget_w, widget_h, tone)
        groups.append({"red": red_poly, "blue": blue})
    return groups


def save_json(file, polygon_groups):
    """儲存紅色與藍色 JSON"""
    if not polygon_groups:
        print(f"跳過 {file}：沒有標註資料")
        return

    red_polys = [g["red"] for g in polygon_groups]
    all_blue_polys = []
    for g in polygon_groups:
        all_blue_polys.extend(g["blue"])
    all_blue_polys = compute_blue_ratios(all_blue_polys)

    red_json, blue_json = get_json_paths(file)

    with open(red_json, "w", encoding="utf-8") as f:
        json.dump({"polys": red_polys}, f, ensure_ascii=False)
    print(f"已儲存 {len(red_polys)} 個紅色多邊形到 {red_json}")

    with open(blue_json, "w", encoding="utf-8") as f:
        json.dump({"polys": all_blue_polys}, f, ensure_ascii=False)
    print(f"已儲存 {len(all_blue_polys)} 個藍色多邊形到 {blue_json}")


# ----------------- Recalc 模式 -----------------
def run_recalc(files):
    for file in files:
        img = read_image(file)
        if img is None:
            continue
        img_show = resize_image(img)

        existing_red_polys, _ = load_existing_json(file)
        if not existing_red_polys:
            print(f"跳過 {file}：無紅色標註可重算")
            continue

        polygon_groups = create_polygon_groups_from_loaded(existing_red_polys, [], img_show.shape)
        print(f"重新計算了 {len(polygon_groups)} 組藍色 poly")
        save_json(file, polygon_groups)

    print("recalc 完成")


# ----------------- Matplotlib 畫圖 -----------------
def redraw(ax, img_show, keep_view=True):
    if keep_view:
        cur_xlim = ax.get_xlim()
        cur_ylim = ax.get_ylim()

    ax.clear()
    ax.imshow(img_show)
    ax.set_aspect("equal")

    if keep_view and cur_xlim and cur_ylim:
        ax.set_xlim(cur_xlim)
        ax.set_ylim(cur_ylim)

    for group in polygon_groups:
        red_poly = group["red"]
        pts = np.array(red_poly["points"])
        color = "lime" if red_poly.get("tone") == "dark" else "g"
        pts_closed = np.vstack([pts, pts[0]])
        ax.plot(pts_closed[:,0], pts_closed[:,1], color=color, linewidth=2)

        for blue_poly in group["blue"]:
            pts_b = np.array(blue_poly["points"])
            if len(pts_b) >= 3:
                pts_closed_b = np.vstack([pts_b, pts_b[0]])
                ax.plot(pts_closed_b[:,0], pts_closed_b[:,1], "b-", linewidth=2)

    if drawing_poly:
        pts = np.array(drawing_poly)
        ax.plot(pts[:,0], pts[:,1], "r-")
        ax.plot(pts[:,0], pts[:,1], "ro")

    ax.set_xticks([]), ax.set_yticks([])
    plt.draw()


def onscroll(event):
    base_scale = 1.2
    ax = event.inaxes
    if ax is None:
        return
    cur_xlim, cur_ylim = ax.get_xlim(), ax.get_ylim()
    xdata, ydata = event.xdata, event.ydata
    if xdata is None or ydata is None:
        return

    scale_factor = 1/base_scale if event.button == "up" else base_scale if event.button == "down" else None
    if scale_factor is None:
        return

    new_width  = (cur_xlim[1] - cur_xlim[0]) * scale_factor
    new_height = (cur_ylim[1] - cur_ylim[0]) * scale_factor
    relx = (xdata - cur_xlim[0]) / (cur_xlim[1] - cur_xlim[0])
    rely = (ydata - cur_ylim[0]) / (cur_ylim[1] - cur_ylim[0])

    ax.set_xlim([xdata - new_width * relx,  xdata + new_width  * (1-relx)])
    ax.set_ylim([ydata - new_height * rely, ydata + new_height * (1-rely)])
    plt.draw()


last_mouse_pos = None

def onmove(event):
    global last_mouse_pos
    if event.xdata is not None and event.ydata is not None:
        last_mouse_pos = (int(event.xdata), int(event.ydata))


def clamp(val, lo, hi):
    return max(lo, min(val, hi))


def onclick(event):
    if event.button == 1 and event.xdata and event.ydata:
        h, w = img_show.shape[:2]
        x, y = clamp(int(event.xdata), 0, w-1), clamp(int(event.ydata), 0, h-1)
        drawing_poly.append((x, y))
        redraw(ax, img_show, keep_view=True)


def onkey(event):
    global drawing_poly, polygon_groups, current_action, last_mouse_pos

    key = event.key

    if key == "q":
        plt.close("all")
        exit()

    elif key == "a":
        if last_mouse_pos:
            h, w = img_show.shape[:2]
            x, y = clamp(last_mouse_pos[0], 0, w-1), clamp(last_mouse_pos[1], 0, h-1)
            drawing_poly.append((x, y))
            redraw(ax, img_show, keep_view=True)

    elif key in ("s", "b"):
        if drawing_poly:
            drawing_poly.pop()
            redraw(ax, img_show, keep_view=True)

    elif key == "n":
        if drawing_poly:
            blue = generate_blue_poly(drawing_poly, img_show.shape, widget_w, widget_h)
            polygon_groups.append({
                "red": {"points": drawing_poly.copy(), "tone": "dark"},
                "blue": blue
            })
            drawing_poly = []
        current_action = "next"
        plt.close()

    elif key == "d":
        if drawing_poly:
            blue = generate_blue_poly(drawing_poly, img_show.shape, widget_w, widget_h, "dark")
            polygon_groups.append({
                "red": {"points": drawing_poly.copy(), "tone": "dark"},
                "blue": blue
            })
            drawing_poly = []
            redraw(ax, img_show, keep_view=False)

    elif key == "l":
        if drawing_poly:
            blue = generate_blue_poly(drawing_poly, img_show.shape, widget_w, widget_h, "light")
            polygon_groups.append({
                "red": {"points": drawing_poly.copy(), "tone": "light"},
                "blue": blue
            })
            drawing_poly = []
            redraw(ax, img_show, keep_view=False)

    elif key == "c":
        if drawing_poly:
            drawing_poly = []
        elif polygon_groups:
            deleted = polygon_groups.pop()
            print(f"已刪除多邊形組：1個紅色多邊形和{len(deleted['blue'])}個藍色多邊形")
        redraw(ax, img_show, keep_view=False)

    elif key == "escape":
        redraw(ax, img_show, keep_view=False)


def on_resize(event):
    plt.tight_layout()
    plt.draw()


# ----------------- 主程式 -----------------
if MODE == "recalc":
    run_recalc(files)
else:
    # label 模式
    for file in files:
        img = read_image(file)
        if img is None:
            continue
        # img 已是 RGB numpy array，不需再 cvtColor
        img_show = resize_image(img)
        drawing_poly = []

        existing_red_polys, existing_blue_polys = load_existing_json(file)
        if existing_red_polys:
            polygon_groups = create_polygon_groups_from_loaded(existing_red_polys, existing_blue_polys, img_show.shape)
            print(f"建立了 {len(polygon_groups)} 個多邊形組")
        else:
            polygon_groups = []
            print("未找到現有標註，開始新的標註")

        current_action = None

        fig, ax = plt.subplots(figsize=(12, 8))
        fig.canvas.mpl_connect("button_press_event", onclick)
        fig.canvas.mpl_connect("key_press_event", onkey)
        fig.canvas.mpl_connect("scroll_event", onscroll)
        fig.canvas.mpl_connect("resize_event", on_resize)
        fig.canvas.mpl_connect("motion_notify_event", onmove)

        redraw(ax, img_show, keep_view=False)
        plt.show()

        save_json(file, polygon_groups)

    print("標註完成，紅色/藍色 JSON 已生成")
