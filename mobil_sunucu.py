import streamlit as st
from ultralytics import YOLO
from PIL import Image
import numpy as np
import cv2
import math
import pillow_heif  # YENİ: Apple formatı eklentisi

# YENİ: Python'un HEIC fotoğrafları açabilmesi için motoru çalıştırır
pillow_heif.register_heif_opener()

st.set_page_config(page_title="Mobil Hibrit Sayım", page_icon="⚙️", layout="centered")
# ... (Modelleri yükleme ve HibritAnalizMotoru kodları aynı kalacak) ...


# Modelleri SADECE BİR KERE belleğe alma (Hız için)
@st.cache_resource
def load_models():
    # Sadece GitHub'a yüklediğimiz kısa dosya adlarını yazıyoruz:
    m_disli = YOLO("disli.pt")
    m_dis = YOLO("dis.pt")
    return m_disli, m_dis

try:
    gear_model, tooth_model = load_models()
except:
    st.error("Modeller yüklenemedi! 'disli_best.pt' ve 'dis_best.pt' dosyalarının proje klasöründe olduğundan emin ol.")


# ================= 4 KRİTERLİ HİBRİT ALGORİTMA MOTORU =================
class HibritAnalizMotoru:
    def _box_area(self, box):
        return max(1.0, (box[3] - box[1]) * (box[4] - box[2]))

    def _intersection_area(self, b1, b2):
        ix1, iy1 = max(b1[1], b2[1]), max(b1[2], b2[2])
        ix2, iy2 = min(b1[3], b2[3]), min(b1[4], b2[4])
        return max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)

    def _filter_duplicate_gears(self, raw_gears):
        if not raw_gears: return []
        sorted_by_conf = sorted(raw_gears, key=lambda x: x[5], reverse=True)
        unique_gears = []

        for g in sorted_by_conf:
            _, gx1, gy1, gx2, gy2, gconf = g
            gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
            g_area = self._box_area(g)
            g_rad = max(gx2 - gx1, gy2 - gy1) / 2.0
            is_duplicate = False

            for idx, ug in enumerate(unique_gears):
                _, ux1, uy1, ux2, uy2, uconf = ug
                ucx, ucy = (ux1 + ux2) / 2.0, (uy1 + uy2) / 2.0
                u_area = self._box_area(ug)
                u_rad = max(ux2 - ux1, uy2 - uy1) / 2.0
                inter = self._intersection_area(g, ug)
                union = g_area + u_area - inter
                iou = inter / max(1.0, union)
                area_ratio = min(g_area, u_area) / max(1.0, max(g_area, u_area))
                center_dist = np.hypot(gcx - ucx, gcy - ucy)

                if iou > 0.45 or (center_dist < 0.30 * max(g_rad, u_rad) and area_ratio > 0.52):
                    nx1, ny1 = min(gx1, ux1), min(gy1, uy1)
                    nx2, ny2 = max(gx2, ux2), max(gy2, uy2)
                    unique_gears[idx] = ["Dişli", nx1, ny1, nx2, ny2, max(gconf, uconf)]
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_gears.append(list(g))
        return unique_gears

    def _fit_circle_ls(self, pts):
        x, y = pts[:, 0], pts[:, 1]
        A = np.column_stack([x, y, np.ones_like(x)])
        b = x ** 2 + y ** 2
        c, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
        xc, yc = c[0] / 2.0, c[1] / 2.0
        r = np.sqrt(max(1.0, c[2] + xc ** 2 + yc ** 2))
        return xc, yc, r

    def _compute_orbit(self, teeth_list, fallback_cx, fallback_cy):
        if len(teeth_list) < 3: return None
        pts = np.array([[(t[1] + t[3]) / 2.0, (t[2] + t[4]) / 2.0] for t in teeth_list])
        angles = np.arctan2(pts[:, 1] - fallback_cy, pts[:, 0] - fallback_cx)
        angles_sorted = np.sort(angles)
        diffs = np.diff(np.concatenate([angles_sorted, [angles_sorted[0] + 2 * np.pi]]))
        max_gap = np.max(diffs)
        angular_span = 2 * np.pi - max_gap

        if angular_span >= 2.6 and len(pts) >= 5:
            xc, yc, r = self._fit_circle_ls(pts)
            if np.hypot(xc - fallback_cx, yc - fallback_cy) < r * 0.6:
                return (xc, yc, r)

        dists = np.hypot(pts[:, 0] - fallback_cx, pts[:, 1] - fallback_cy)
        return (fallback_cx, fallback_cy, float(np.median(dists)))

    def assign_teeth_four_criteria(self, raw_teeth, raw_gears):
        unique_gears = self._filter_duplicate_gears(raw_gears)
        sorted_gears = sorted(unique_gears, key=self._box_area)
        num_gears = len(sorted_gears)
        if num_gears == 0: return [], raw_teeth

        parent_of = {}
        children_of = {i: [] for i in range(num_gears)}

        for i in range(num_gears):
            area_i = self._box_area(sorted_gears[i])
            icx, icy = (sorted_gears[i][1] + sorted_gears[i][3]) / 2.0, (sorted_gears[i][2] + sorted_gears[i][4]) / 2.0
            for j in range(i + 1, num_gears):
                area_j = self._box_area(sorted_gears[j])
                jcx, jcy = (sorted_gears[j][1] + sorted_gears[j][3]) / 2.0, (
                            sorted_gears[j][2] + sorted_gears[j][4]) / 2.0
                jrx, jry = max(1.0, (sorted_gears[j][3] - sorted_gears[j][1]) / 2.0), max(1.0, (
                            sorted_gears[j][4] - sorted_gears[j][2]) / 2.0)
                inter_ratio = self._intersection_area(sorted_gears[i], sorted_gears[j]) / area_i
                e_center = np.sqrt(((icx - jcx) / jrx) ** 2 + ((icy - jcy) / jry) ** 2)
                if area_i < 0.55 * area_j and inter_ratio >= 0.65 and e_center <= 0.65:
                    parent_of[i] = j
                    children_of[j].append(i)
                    break

        gear_teeth = {i: [] for i in range(num_gears)}
        assigned_tooth_indices = set()

        for t_idx, t_box in enumerate(raw_teeth):
            tcx, tcy = (t_box[1] + t_box[3]) / 2.0, (t_box[2] + t_box[4]) / 2.0
            assigned_concentric = False
            for p_idx, c_list in children_of.items():
                if not c_list: continue
                _, px1, py1, px2, py2, _ = sorted_gears[p_idx]
                pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                prx, pry = max(1.0, (px2 - px1) / 2.0), max(1.0, (py2 - py1) / 2.0)
                if np.sqrt(((tcx - pcx) / prx) ** 2 + ((tcy - pcy) / pry) ** 2) < 0.74:
                    best_c = min(c_list, key=lambda c: np.hypot(tcx - (sorted_gears[c][1] + sorted_gears[c][3]) / 2.0,
                                                                tcy - (sorted_gears[c][2] + sorted_gears[c][4]) / 2.0))
                    gear_teeth[best_c].append((t_idx, t_box))
                    assigned_tooth_indices.add(t_idx)
                    assigned_concentric = True
                    break

            if assigned_concentric: continue
            candidates = []
            for g_idx, g_box in enumerate(sorted_gears):
                _, gx1, gy1, gx2, gy2, _ = g_box
                gcx, gcy = (gx1 + gx2) / 2.0, (gy1 + gy2) / 2.0
                grx, gry = max(1.0, (gx2 - gx1) / 2.0), max(1.0, (gy2 - gy1) / 2.0)
                e_dist = np.sqrt(((tcx - gcx) / grx) ** 2 + ((tcy - gcy) / gry) ** 2)

                if g_idx in parent_of:
                    p_idx = parent_of[g_idx]
                    _, px1, py1, px2, py2, _ = sorted_gears[p_idx]
                    pcx, pcy = (px1 + px2) / 2.0, (py1 + py2) / 2.0
                    prx, pry = max(1.0, (px2 - px1) / 2.0), max(1.0, (py2 - py1) / 2.0)
                    if np.sqrt(((tcx - pcx) / prx) ** 2 + ((tcy - pcy) / pry) ** 2) >= 0.74: continue

                if e_dist <= 1.08:
                    candidates.append((abs(e_dist - 0.96), g_idx))

            if candidates:
                candidates.sort(key=lambda x: x[0])
                gear_teeth[candidates[0][1]].append((t_idx, t_box))
                assigned_tooth_indices.add(t_idx)

        orbits = {}
        for g_idx, g_box in enumerate(sorted_gears):
            ref_cx, ref_cy = (g_box[1] + g_box[3]) / 2.0, (g_box[2] + g_box[4]) / 2.0
            if g_idx in parent_of:
                p_box = sorted_gears[parent_of[g_idx]]
                ref_cx, ref_cy = (p_box[1] + p_box[3]) / 2.0, (p_box[2] + p_box[4]) / 2.0

            orbit = self._compute_orbit([item[1] for item in gear_teeth[g_idx]], ref_cx, ref_cy)
            orbits[g_idx] = orbit

            if orbit is not None and len(gear_teeth[g_idx]) >= 7:
                ocx, ocy, o_r = orbit
                med_size = float(
                    np.median([max(item[1][3] - item[1][1], item[1][4] - item[1][2]) for item in gear_teeth[g_idx]]))
                tol = max(med_size * 1.8, o_r * 0.22)
                kept = []
                for t_idx, t_box in gear_teeth[g_idx]:
                    tcx, tcy = (t_box[1] + t_box[3]) / 2.0, (t_box[2] + t_box[4]) / 2.0
                    if abs(np.hypot(tcx - ocx, tcy - ocy) - o_r) <= tol:
                        kept.append((t_idx, t_box))
                    else:
                        assigned_tooth_indices.discard(t_idx)
                gear_teeth[g_idx] = kept
                orbits[g_idx] = self._compute_orbit([x[1] for x in kept], ref_cx, ref_cy)

        groups = []
        for g_idx, g_box in enumerate(sorted_gears):
            groups.append(([f"Dişli #{g_idx + 1}", g_box[1], g_box[2], g_box[3], g_box[4], g_box[5]],
                           [x[1] for x in gear_teeth[g_idx]], orbits.get(g_idx)))
        rejected = [raw_teeth[i] for i in range(len(raw_teeth)) if i not in assigned_tooth_indices]

        return groups, rejected


# ================= ARAYÜZ (WEB/MOBİL) =================
st.info("💡 Telefonunuzdan kamerayı açmak veya fotoğraf yüklemek için dokunun.")
# YENİ:
uploaded_file = st.file_uploader("", type=["jpg", "jpeg", "png", "heic", "HEIC"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📸 Çekilen Fotoğraf", use_container_width=True)

    # Mobildeki buton
    if st.button("🚀 4 Kriterli Hibrit Analizi Başlat", use_container_width=True):
        with st.spinner('Matematiksel Filtreler ve YOLO Analizi Çalışıyor...'):

            # 1. Ham YOLO Tahminleri
            raw_gears, raw_teeth = [], []
            for r in gear_model.predict(image, imgsz=1280, conf=0.15, verbose=False):
                for b in r.boxes:
                    gx1, gy1, gx2, gy2 = map(float, b.xyxy[0])
                    raw_gears.append(["Dişli", gx1, gy1, gx2, gy2, float(b.conf[0])])

            for r in tooth_model.predict(image, imgsz=1280, conf=0.35, max_det=2000, verbose=False):
                for b in r.boxes:
                    tx1, ty1, tx2, ty2 = map(float, b.xyxy[0])
                    raw_teeth.append(["Diş", tx1, ty1, tx2, ty2, float(b.conf[0]), 1])

            # 2. HİBRİT MOTORU ÇALIŞTIR
            motor = HibritAnalizMotoru()
            groups, rejected = motor.assign_teeth_four_criteria(raw_teeth, raw_gears)

            # 3. OpenCV İle Sonuç Çizimi (Tkinter yerine CV2)
            img_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)

            toplam_sayilan = 0
            for g_box, t_list, orbit_data in groups:
                g_label, gx1, gy1, gx2, gy2, _ = g_box

                # Dişli Gövdesi Çizimi (Mor Kesik Çizgi efekti yerine düz mor çizgi)
                cv2.rectangle(img_cv, (int(gx1), int(gy1)), (int(gx2), int(gy2)), (252, 132, 192), 3)
                cv2.putText(img_cv, f"{g_label}: {len(t_list)} Dis", (int(gx1), max(10, int(gy1) - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                toplam_sayilan += len(t_list)

                # Yörünge (Sarı Daire)
                if orbit_data is not None:
                    xc, yc, r_orb = orbit_data
                    cv2.circle(img_cv, (int(xc), int(yc)), int(r_orb), (0, 204, 255), 2)
                    cv2.circle(img_cv, (int(xc), int(yc)), 5, (0, 0, 255), -1)  # Merkez nokta

                # Gruba dahil edilen dişler (Açık Mavi)
                for t_box in t_list:
                    _, tx1, ty1, tx2, ty2, _, _ = t_box
                    cv2.rectangle(img_cv, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (248, 189, 56), 2)

            # Reddedilen gürültü dişler (Kırmızı)
            for t_box in rejected:
                _, tx1, ty1, tx2, ty2, _, _ = t_box
                cv2.rectangle(img_cv, (int(tx1), int(ty1)), (int(tx2), int(ty2)), (0, 0, 255), 2)

            # Görüntüyü geri çevirip ekrana bas
            final_image = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
            st.image(final_image, caption="Hibrit Analiz Sonucu", use_container_width=True)

            # Raporlama Ekranı
            st.success(f"✅ TOPLAM {len(groups)} DİŞLİ KATMANI TESPİT EDİLDİ")
            for g_box, t_list, _ in groups:
                st.write(f"▶ **{g_box[0]}**: {len(t_list)} Adet Diş (Mavi Kutu)")

            if len(rejected) > 0:
                st.warning(f"⚠️ Zincir Dışı / Gürültü Olarak Reddedilen: {len(rejected)} Adet (Kırmızı Kutu)")