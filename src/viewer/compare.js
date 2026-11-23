import * as THREE from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { SparkRenderer, SplatMesh } from '@sparkjsdev/spark';

const $ = id => document.getElementById(id);
async function json(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(`${response.status}: ${url}`);
  return response.json();
}
async function main() {
  const query = new URLSearchParams(location.search);
  const scene = query.get('scene'), object = query.get('object'), version = query.get('version');
  const path = [version, scene, object].map(encodeURIComponent).join('/');
  const base = `/result/${path}/`;
  $('title').textContent = `${scene} / ${object}`;
  const [frame, transform] = await Promise.all([json(`/api/frame/${path}`), json(base + 'transform.json')]);
  const rodinToWorld = new THREE.Matrix4().fromArray(transform.rodinToWorld.matrix);
  const worldToView = new THREE.Matrix4().fromArray(frame.canonicalToWorld).invert();
  const viewport = $('viewport');
  const gRenderer = new THREE.WebGLRenderer({canvas:$('gaussian'),antialias:true});
  const mRenderer = new THREE.WebGLRenderer({canvas:$('model'),antialias:true,alpha:true});
  for (const renderer of [gRenderer, mRenderer]) {
    renderer.setPixelRatio(Math.min(devicePixelRatio, 2));
    renderer.outputColorSpace = THREE.SRGBColorSpace;
    renderer.toneMapping = THREE.NoToneMapping;
  }
  const gScene = new THREE.Scene(), mScene = new THREE.Scene();
  gScene.add(new SparkRenderer({renderer:gRenderer}));
  mScene.add(new THREE.HemisphereLight(0xffffff, 0x7b8390, 2));
  const light = new THREE.DirectionalLight(0xffffff, 2); light.position.set(3, 5, 4); mScene.add(light);
  const material = new THREE.MeshStandardMaterial({color:0x718fb5,roughness:.85,side:THREE.DoubleSide});
  const splat = new SplatMesh({url:base + 'gaussian_filter.ply',fileName:'gaussian.ply'});
  worldToView.decompose(splat.position, splat.quaternion, splat.scale);
  const loader = new GLTFLoader();
  const [gltf] = await Promise.all([loader.loadAsync(base + 'rodin.glb'), splat.initialized]);
  const model = new THREE.Group();
  model.matrix.multiplyMatrices(worldToView, rodinToWorld); model.matrixAutoUpdate = false;
  model.add(gltf.scene);
  gltf.scene.traverse(node => {
    if (node.isMesh) {
      node.material = material;
      if (!node.geometry.attributes.normal) node.geometry.computeVertexNormals();
    }
  });
  gScene.add(splat); mScene.add(model);
  const camera = new THREE.PerspectiveCamera(35, 1, .001, 1000);
  const controls = new OrbitControls(camera, viewport); controls.enableDamping = true;
  const target = new THREE.Vector3(...frame.obb.center).applyMatrix4(worldToView);
  const scale = new THREE.Vector3().setFromMatrixScale(worldToView).x;
  const radius = new THREE.Vector3(...frame.obb.size).length() * scale / 2;
  camera.near = radius / 1000; camera.far = radius * 100;
  function view(value) {
    const angle = Number(value) * Math.PI / 180;
    const direction = value === 'top' ? new THREE.Vector3(0, 1, .0001)
      : new THREE.Vector3(Math.sin(angle) * Math.cos(Math.PI / 6), .5, Math.cos(angle) * Math.cos(Math.PI / 6));
    const aspect = viewport.clientWidth / viewport.clientHeight / ($('layout').value === 'side' ? 2 : 1);
    const halfFov = Math.min(camera.fov * Math.PI / 360, Math.atan(Math.tan(camera.fov * Math.PI / 360) * aspect));
    controls.target.copy(target); camera.position.copy(target).addScaledVector(direction, radius / Math.sin(halfFov) * 1.12);
    controls.update();
  }
  document.querySelectorAll('[data-view]').forEach(button => button.onclick = () => view(button.dataset.view));
  $('layout').onchange = () => {
    const overlay = $('layout').value === 'overlay';
    material.transparent = overlay; material.opacity = overlay ? .55 : 1.; material.needsUpdate = true;
  };
  view('0'); $('status').hidden = true;
  window.viewerState = () => ({rodinToWorld:rodinToWorld.toArray(), gaussianMatrix:splat.matrix.toArray(),
    modelMatrix:model.matrix.toArray(), displaySplats:splat.packedSplats.numSplats, layout:$('layout').value});
  gRenderer.setAnimationLoop(() => {
    const layout = $('layout').value, width = viewport.clientWidth, height = viewport.clientHeight;
    controls.update(); camera.updateMatrixWorld();
    camera.aspect = width / height / (layout === 'side' ? 2 : 1); camera.updateProjectionMatrix();
    for (const [renderer, source, isGaussian] of [[gRenderer,gScene,true],[mRenderer,mScene,false]]) {
      const canvas = renderer.domElement;
      const hidden = (layout === 'gaussian' && !isGaussian) || (layout === 'model' && isGaussian);
      canvas.style.display = hidden ? 'none' : 'block';
      canvas.style.width = layout === 'side' ? '50%' : '100%';
      canvas.style.left = layout === 'side' && !isGaussian ? '50%' : '0';
      if (hidden) continue;
      const viewWidth = layout === 'side' ? width / 2 : width, pixelRatio = renderer.getPixelRatio();
      if (canvas.width !== Math.floor(viewWidth * pixelRatio) || canvas.height !== Math.floor(height * pixelRatio)) renderer.setSize(viewWidth, height, false);
      renderer.setClearColor(0xffffff, !isGaussian && layout === 'overlay' ? 0 : 1);
      renderer.render(source, camera);
    }
    document.body.dataset.ready = 'true';
  });
}
main().catch(error => { $('status').hidden = false; $('status').textContent = error.message; document.body.dataset.ready = 'error'; console.error(error); });
