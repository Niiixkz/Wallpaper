for d in [0-9]*; do
    for f in "$d"/H.{png,jpg}; do
        [ -f "$f" ] || continue
        avifenc --yuv 444 -q 90 -s 0 --ignore-exif --ignore-xmp "$f" "$d/H.avif"
        rm "$f"
    done
done

for d in [0-9]*; do
    for f in "$d"/V.{png,jpg}; do
        [ -f "$f" ] || continue
        avifenc --yuv 444 -q 90 -s 0 --ignore-exif --ignore-xmp "$f" "$d/V.avif"
        rm "$f"
    done
done
