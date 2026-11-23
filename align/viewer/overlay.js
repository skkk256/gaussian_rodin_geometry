import * as THREE from 'three';
import {OrbitControls} from 'three/addons/controls/OrbitControls.js';
import {GLTFLoader} from 'three/addons/loaders/GLTFLoader.js';
import {SparkRenderer,SplatMesh} from '@sparkjsdev/spark';

export class OverlayViewer {
  constructor(viewport,onStatus) {
    this.viewport=viewport;this.onStatus=onStatus;this.layers=[];this.pending=0;this.animation=0;this.drawCount=0;this.meshParseCount=0;
    this.requestRender=()=>{if(!this.animation)this.animation=requestAnimationFrame(()=>this.draw());};
    this.meshCanvas=document.createElement('canvas');this.meshCanvas.id='mesh-canvas';
    this.gsCanvas=document.createElement('canvas');this.gsCanvas.id='gs-canvas';
    viewport.prepend(this.gsCanvas,this.meshCanvas);
    this.renderer=new THREE.WebGLRenderer({canvas:this.meshCanvas,alpha:true,antialias:true});
    this.gsRenderer=new THREE.WebGLRenderer({canvas:this.gsCanvas,alpha:true,antialias:false});
    for(const renderer of [this.renderer,this.gsRenderer]){renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));renderer.outputColorSpace=THREE.SRGBColorSpace;renderer.setClearColor(0x26313c,0);}
    this.scene=new THREE.Scene();this.scene.add(new THREE.HemisphereLight(0xffffff,0x596a82,2));
    const light=new THREE.DirectionalLight(0xffffff,3);light.position.set(3,5,4);this.scene.add(light);
    this.gsScene=new THREE.Scene();this.spark=new SparkRenderer({renderer:this.gsRenderer,onDirty:this.requestRender});this.gsScene.add(this.spark);
    this.camera=new THREE.PerspectiveCamera(35,1,.001,100);this.camera.up.set(0,-1,0);
    this.controls=new OrbitControls(this.camera,viewport);this.controls.addEventListener('change',this.requestRender);
    this.resizeObserver=new ResizeObserver(this.requestRender);this.resizeObserver.observe(viewport);
    this.material='color';this.pointSize=2.5;this.pointXray=true;this.wireframe=false;
    this.loader=new GLTFLoader();
    this.geometryLoader=new GLTFLoader().register(parser=>({name:'GeometryWithoutTextures',beforeRoot(){for(const mesh of parser.json.meshes)for(const primitive of mesh.primitives)delete primitive.material;}}));
  }
  status() {
    this.onStatus(this.pending,this.layers.filter(l=>l.visible).map(l=>l.label));this.requestRender();
  }
  draw() {
    this.animation=0;this.drawCount++;
    const w=this.viewport.clientWidth,h=this.viewport.clientHeight,pixelRatio=this.renderer.getPixelRatio();
    this.camera.aspect=w/h;this.camera.updateProjectionMatrix();this.camera.updateMatrixWorld();
    for(const renderer of [this.renderer,this.gsRenderer])if(renderer.domElement.width!==Math.round(w*pixelRatio)||renderer.domElement.height!==Math.round(h*pixelRatio))renderer.setSize(w,h,false);
    const gsVisible=this.layers.some(l=>l.kind==='gs'&&l.visible&&l.object);
    this.gsCanvas.style.display=gsVisible?'block':'none';
    if(gsVisible)this.gsRenderer.render(this.gsScene,this.camera);
    this.renderer.render(this.scene,this.camera);
    if(this.pending||this.spark.sorting)this.requestRender();
  }
  reset() {
    const geometries=new Set(),materials=new Set(),textures=new Set();
    for(const l of this.layers){
      if(l.object){l.object.removeFromParent();if(l.kind==='gs')l.object.dispose();else l.object.traverse(o=>{if(o.geometry)geometries.add(o.geometry);if(o.material)materials.add(o.material);if(o.userData.materials)for(const material of Object.values(o.userData.materials))materials.add(material);});}
    }
    for(const material of materials){for(const value of Object.values(material))if(value?.isTexture)textures.add(value);material.dispose();}
    for(const geometry of geometries)geometry.dispose();
    for(const texture of textures){texture.dispose();texture.source.data?.close?.();}
    this.layers=[];this.buffer=null;this.geometry=null;this.originalMaterial=null;this.sourceUrl=null;this.spark.setDirty();
  }
  async setCase(layers,meshUrl,bounds,orientation) {
    this.reset();this.layers=layers;this.sourceUrl=meshUrl;this.orientation=orientation;
    this.center=new THREE.Vector3(...bounds[0]).add(new THREE.Vector3(...bounds[1])).multiplyScalar(.5);
    this.radius=new THREE.Vector3(...bounds[1]).sub(new THREE.Vector3(...bounds[0])).length()/2;
    this.view('45');
    await Promise.all(layers.filter(l=>l.visible).map(l=>this.load(l)));
    this.status();
  }
  view(value) {
    const angle=Number(value)*Math.PI/180;
    const direction=value==='top'?new THREE.Vector3(0,1,.0001):new THREE.Vector3(Math.sin(angle),.38,Math.cos(angle));
    direction.transformDirection(this.orientation);
    const aspect=this.viewport.clientWidth/this.viewport.clientHeight;
    const fov=Math.min(this.camera.fov*Math.PI/360,Math.atan(Math.tan(this.camera.fov*Math.PI/360)*aspect));
    this.camera.near=this.radius/1000;this.camera.far=this.radius*150;
    this.controls.target.copy(this.center);this.camera.position.copy(this.center).addScaledVector(direction,this.radius/Math.sin(fov)*1.05);this.controls.update();this.requestRender();
  }
  frame() {
    const box=new THREE.Box3();this.scene.updateMatrixWorld(true);
    for(const l of this.layers.filter(l=>l.visible&&l.object)){
      if(l.kind==='gs')box.union(new THREE.Box3(new THREE.Vector3(...l.bounds[0]),new THREE.Vector3(...l.bounds[1])).applyMatrix4(l.display));
      else box.union(new THREE.Box3().setFromObject(l.object));
    }
    if(!box.isEmpty()){box.getCenter(this.center);this.radius=box.getSize(new THREE.Vector3()).length()/2;this.view('45');}
  }
  async geometryScene() {
    this.buffer??=fetch(this.sourceUrl).then(r=>{if(!r.ok)throw Error(`${r.status}: ${this.sourceUrl}`);return r.arrayBuffer();});
    this.geometry??=this.buffer.then(buffer=>this.geometryLoader.parseAsync(buffer,'')).then(gltf=>{this.meshParseCount++;return gltf.scene;});
    return this.geometry;
  }
  async texturedMaterial() {
    this.originalMaterial??=this.buffer.then(buffer=>this.loader.parseAsync(buffer,'')).then(gltf=>{
      let material;gltf.scene.traverse(o=>{if(o.isMesh){material=o.material;o.geometry.dispose();}});return material;
    });
    return this.originalMaterial;
  }
  style(l) {
    if(!l.object)return;
    l.object.visible=l.visible;
    if(l.kind==='gs'){l.object.opacity=l.opacity;this.spark.setDirty();}
    else if(l.kind==='points'){l.object.material.opacity=l.opacity;l.object.material.size=this.pointSize;l.object.material.depthTest=!this.pointXray;}
    else l.object.traverse(o=>{if(o.isMesh){const material=o.userData.materials[this.material];o.material=material;material.opacity=l.opacity;material.transparent=l.opacity<1;material.depthWrite=l.opacity===1||!this.pointXray;material.wireframe=this.wireframe;material.side=THREE.DoubleSide;material.forceSinglePass=true;material.needsUpdate=true;}});
    this.requestRender();
  }
  async load(l) {
    if(l.promise)return l.promise;
    l.promise=(async()=>{
      this.pending++;this.status();
      try{
        if(l.kind==='gs'){
          const o=new SplatMesh({url:l.url,fileName:'gaussian.ply'});await o.initialized;l.object=o;
          o.matrixAutoUpdate=false;o.matrix.copy(l.display);this.gsScene.add(o);this.spark.setDirty();
          this.gsScene.updateMatrixWorld(true);this.camera.updateMatrixWorld(true);
          await this.spark.update({scene:this.gsScene,camera:this.camera});
          this.requestRender();
        }else if(l.kind==='points'){
          const response=await fetch(l.url);if(!response.ok)throw Error(`${response.status}: ${l.url}`);
          const points=await response.json(),geometry=new THREE.BufferGeometry();geometry.setAttribute('position',new THREE.Float32BufferAttribute(points.flat(),3));
          l.object=new THREE.Points(geometry,new THREE.PointsMaterial({color:l.color,sizeAttenuation:false,transparent:true,depthWrite:false}));
          l.object.matrixAutoUpdate=false;l.object.matrix.copy(l.display);l.object.renderOrder=10;this.scene.add(l.object);
        }else{
          l.object=(await this.geometryScene()).clone(true);
          const node=l.object.children[0];node.matrixAutoUpdate=false;node.matrix.fromArray(l.matrix);
          l.object.matrixAutoUpdate=false;l.object.matrix.copy(l.display);
          l.object.traverse(o=>{if(o.isMesh){o.userData={materials:{color:new THREE.MeshStandardMaterial({color:l.color,roughness:.85})}};}});
          if(this.material==='original'){const material=await this.texturedMaterial();l.object.traverse(o=>{if(o.isMesh)o.userData.materials.original=material.clone();});}
          this.scene.add(l.object);
        }
        this.style(l);
      }finally{this.pending--;this.status();}
    })();return l.promise;
  }
  async setVisible(ids) {
    for(const l of this.layers){l.visible=ids.includes(l.id);if(l.object)this.style(l);}
    await Promise.all(this.layers.filter(l=>l.visible).map(l=>this.load(l)));this.status();
  }
  async setMaterial(value) {
    if(value==='original'){
      this.pending++;this.status();
      try{
        await this.geometryScene();const material=await this.texturedMaterial();
        for(const l of this.layers.filter(l=>l.kind==='mesh'&&l.object))l.object.traverse(o=>{if(o.isMesh&&!o.userData.materials.original)o.userData.materials.original=material.clone();});
      }finally{this.pending--;this.status();}
    }
    this.material=value;for(const l of this.layers)this.style(l);
  }
  state() {
    return {meshParseCount:this.meshParseCount,drawCount:this.drawCount,pending:this.pending,triangles:this.renderer.info.render.triangles,geometries:this.renderer.info.memory.geometries,textures:this.renderer.info.memory.textures,
      camera:this.camera.position.toArray(),target:this.controls.target.toArray(),layers:this.layers.map(l=>({id:l.id,visible:l.visible,loaded:!!l.object,opacity:l.opacity,display:l.display.elements,matrix:l.matrix,pointCount:l.kind==='points'&&l.object?l.object.geometry.attributes.position.count:null,splats:l.kind==='gs'?l.object?.numSplats:null,geometry:l.kind==='mesh'&&l.object?l.object.children[0].geometry.uuid:null}))};
  }
}
