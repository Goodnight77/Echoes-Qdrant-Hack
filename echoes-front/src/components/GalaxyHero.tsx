import { useEffect, useRef } from 'react';
import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { api, type VizItem } from '@/lib/api';

const ACCENT = new THREE.Color('#A78BFA');
const TYPE_COLORS: Record<string, THREE.Color> = {
  photo: new THREE.Color('#60a5fa'),
  video: new THREE.Color('#f472b6'),
  voice_memo: new THREE.Color('#fbbf24'),
  screenshot: new THREE.Color('#34d399'),
  unknown: new THREE.Color('#a1a1aa'),
};

const SPHERE_GEO = new THREE.SphereGeometry(0.14, 14, 12);

export default function GalaxyHero() {
  const containerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    const el = containerRef.current;
    if (!el) return;

    const w = el.clientWidth, h = el.clientHeight;
    const scene = new THREE.Scene();
    scene.fog = new THREE.FogExp2('#06060a', 0.03);

    const camera = new THREE.PerspectiveCamera(45, w / h, 0.1, 200);
    camera.position.set(8, 5, 10);

    const renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    renderer.setSize(w, h, false);
    renderer.setClearColor(0, 0);
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    el.appendChild(renderer.domElement);

    scene.add(new THREE.AmbientLight(0xffffff, 0.55));
    const key = new THREE.DirectionalLight(0xffffff, 0.8);
    key.position.set(5, 8, 6);
    scene.add(key);
    const rim = new THREE.PointLight('#A78BFA', 22, 30, 1.8);
    rim.position.set(-6, 4, 0);
    scene.add(rim);

    // Stars
    const starCount = 400;
    const starPos = new Float32Array(starCount * 3);
    for (let i = 0; i < starCount; i++) {
      const r = 30 + Math.random() * 30;
      const theta = Math.random() * Math.PI * 2;
      const phi = Math.acos(2 * Math.random() - 1);
      starPos[i * 3] = r * Math.sin(phi) * Math.cos(theta);
      starPos[i * 3 + 1] = r * Math.sin(phi) * Math.sin(theta);
      starPos[i * 3 + 2] = r * Math.cos(phi);
    }
    const starGeo = new THREE.BufferGeometry();
    starGeo.setAttribute('position', new THREE.BufferAttribute(starPos, 3));
    const stars = new THREE.Points(starGeo, new THREE.PointsMaterial({
      color: 0xffffff, size: 0.05, transparent: true, opacity: 0.5,
    }));
    scene.add(stars);

    const controls = new OrbitControls(camera, renderer.domElement);
    controls.enableDamping = true;
    controls.dampingFactor = 0.08;
    controls.autoRotate = true;
    controls.autoRotateSpeed = 0.5;
    controls.minDistance = 3;
    controls.maxDistance = 30;
    controls.target.set(0, 0, 0);

    let pointGroup: THREE.Group | null = null;
    const pickables: THREE.Mesh[] = [];

    // Load projection
    api.vizProjection('visual', 3).then(data => {
      const items = data.items || [];
      if (!items.length) return;

      pointGroup = new THREE.Group();
      const edgePositions: number[] = [];
      const idIndex = new Map<string, number>();
      items.forEach((it, i) => idIndex.set(it.id, i));

      for (const it of items) {
        const color = TYPE_COLORS[it.type] || TYPE_COLORS.unknown;
        const mat = new THREE.MeshStandardMaterial({
          color, roughness: 0.4, metalness: 0.1, emissive: color, emissiveIntensity: 0.3,
        });
        const mesh = new THREE.Mesh(SPHERE_GEO, mat);
        mesh.position.set(it.x, it.y, it.z);
        (mesh as any).userData = { item: it };
        pointGroup!.add(mesh);
        pickables.push(mesh);

        // Edges to neighbors
        for (const nid of it.neighbor_ids || []) {
          const ni = idIndex.get(nid);
          if (ni !== undefined) {
            edgePositions.push(it.x, it.y, it.z, items[ni].x, items[ni].y, items[ni].z);
          }
        }
      }

      if (edgePositions.length) {
        const edgeGeo = new THREE.BufferGeometry();
        edgeGeo.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
        const edgeLine = new THREE.LineSegments(edgeGeo, new THREE.LineBasicMaterial({
          color: 0xA78BFA, transparent: true, opacity: 0.12,
        }));
        pointGroup.add(edgeLine);
      }

      scene.add(pointGroup);
    });

    let raf: number;
    const tick = () => {
      controls.update();
      renderer.render(scene, camera);
      raf = requestAnimationFrame(tick);
    };
    tick();

    const onResize = () => {
      const W = el.clientWidth, H = el.clientHeight;
      renderer.setSize(W, H, false);
      camera.aspect = W / H;
      camera.updateProjectionMatrix();
    };
    window.addEventListener('resize', onResize);

    return () => {
      cancelAnimationFrame(raf);
      window.removeEventListener('resize', onResize);
      renderer.dispose();
      controls.dispose();
      scene.clear();
      if (el.contains(renderer.domElement)) el.removeChild(renderer.domElement);
    };
  }, []);

  return (
    <div
      ref={containerRef}
      className="relative mb-6 rounded-2xl overflow-hidden border border-[#1e1e28] bg-gradient-to-b from-[#0d0d18] to-[#06060a]"
      style={{ height: 420 }}
    />
  );
}
