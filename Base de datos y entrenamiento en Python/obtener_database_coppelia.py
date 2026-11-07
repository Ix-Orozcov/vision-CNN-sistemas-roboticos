"""
Genera la base de datos COMPLETA (train / validation / test) en CoppeliaSim, lista para la CNN:

  1. Captura las 4 clases (avanzar, izquierda, derecha, obstaculo) con la cámara virtual,
     variando distancia, posición del patrón y AHORA TAMBIÉN el ángulo de la cámara
     (giro, inclinación, rotación y altura).
  2. Redimensiona cada foto a 64 x 64 (Lanczos).
  3. Aplica ruido (gaussiano + sal y pimienta) a exactamente el 20 % de las imágenes
     de cada clase en cada grupo, igual que el script agregar_ruido.py.

Estructura de salida:
    dataset_run/
        train/       avanzar/ izquierda/ derecha/ obstaculo/
        validation/  avanzar/ izquierda/ derecha/ obstaculo/
        test/        avanzar/ izquierda/ derecha/ obstaculo/
        manifest.csv      (parámetros de cada imagen, incluye si lleva ruido)
        vista_previa.png
        config.json
        escena_dataset.ttt

Uso (CoppeliaSim abierto, escena NUEVA y simulación detenida):
    python capturar_dataset.py --output dataset_run
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

CLASSES = ['avanzar', 'izquierda', 'derecha', 'obstaculo']
# Symbols are defined as seen by the camera: image right = world -Y.
PATTERNS = {
 'avanzar': ['0001000','0011100','0111110','0001000','0001000','0001000','0001000'],
 'izquierda': ['0001000','0011000','0111111','1111111','0111111','0011000','0001000'],
 'derecha': ['0001000','0001100','1111110','1111111','1111110','0001100','0001000'],
}

# ---------------- Variación del ángulo de cámara (radianes / metros) ----------------
YAW_MAX = .18      # giro a izquierda/derecha   (±10°)
PITCH_MAX = .15    # inclinación arriba/abajo   (±9°)
ROLL_MAX = .25     # rotación sobre su eje      (±14°)
CAM_Z = (.32, .62) # altura de la cámara

# ---------------- Ruido (mismos valores que agregar_ruido.py) ----------------
SIGMA = 0.05       # ruido gaussiano (en escala 0-1)
PROB_SP = 0.02     # 2 % de píxeles con sal y pimienta


def split_for(group):
    return 'train' if group < 14 else ('validation' if group < 17 else 'test')

def camera_matrix(x, y, z, yaw):
    # Columns: sensor right, sensor up, sensor forward. Forward is local +Z.
    c, s = math.cos(yaw), math.sin(yaw)
    return [s,0,c,x, -c,0,s,y, 0,1,0,z]

def main():
    p = argparse.ArgumentParser(description='Construye una escena y captura el dataset completo con cámara virtual.')
    p.add_argument('--output', default='dataset_run')
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=23000)
    p.add_argument('--seed', type=int, default=73)
    p.add_argument('--frames-per-group', type=int, default=25)
    p.add_argument('--resolution', type=int, default=320)
    p.add_argument('--save-size', type=int, default=64,
                   help='Tamaño final de las imágenes (0 = guardar a la resolución de captura).')
    p.add_argument('--noise-fraction', type=float, default=0.20,
                   help='Fracción de imágenes con ruido (0.20 = 20 %%).')
    p.add_argument('--build-only', action='store_true')
    args = p.parse_args()
    if args.frames_per_group < 1 or args.resolution < 64:
        p.error('Usa al menos 1 frame por grupo y resolución >=64.')
    if not 0 <= args.noise_fraction <= 1:
        p.error('--noise-fraction debe estar entre 0 y 1.')
    out = Path(args.output).resolve()
    if out.exists():
        p.error('La carpeta de salida ya existe. Usa un nombre nuevo para evitar sobrescrituras.')
    from coppeliasim_zmqremoteapi_client import RemoteAPIClient
    import numpy as np
    from PIL import Image, ImageDraw
    client = RemoteAPIClient(host=args.host, port=args.port)
    sim = client.require('sim')
    if sim.getSimulationState() != sim.simulation_stopped:
        raise RuntimeError('Detén la simulación antes de ejecutar.')
    # Do not clear an existing user scene.
    if sim.getObjectsInTree(sim.handle_scene, sim.object_shape_type, 0):
        raise RuntimeError('Usa una escena nueva y elimina suelo/modelos existentes. Se requieren cero shapes.')
    out.mkdir(parents=True)
    rng = random.Random(args.seed)
    rng_ruido = np.random.default_rng(args.seed)
    world = sim.handle_world

    def agregar_ruido(image):
        """Ruido gaussiano + sal y pimienta, igual que agregar_ruido.py (TIPO_RUIDO = 'ambos')."""
        arr = np.asarray(image, dtype=np.float32) / 255.0
        arr = arr + rng_ruido.normal(0, SIGMA, arr.shape)
        mascara = rng_ruido.random(arr.shape[:2])          # misma máscara en los 3 canales
        arr[mascara < PROB_SP / 2] = 0.0
        arr[mascara > 1 - PROB_SP / 2] = 1.0
        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255).round().astype(np.uint8))

    def color(h, rgb):
        sim.setShapeColor(h, None, sim.colorcomponent_ambient_diffuse, rgb)
    def box(name, size, pos, rgb, parent=None):
        h = sim.createPrimitiveShape(sim.primitiveshape_cuboid, size, 0)
        sim.setObjectAlias(h, name)
        sim.setObjectInt32Param(h, sim.shapeintparam_static, 1)
        sim.setObjectPosition(h, world, pos)
        color(h, rgb)
        if parent is not None:
            sim.setObjectParent(h, parent, True)
        return h
    floor = box('Suelo', [8,6,.1], [1,0,-.05], [.48,.48,.48])
    back = box('Fondo', [.1,6,2.5], [4,0,1.25], [.7,.7,.7])
    robot = box('RobotAdquisicion', [.35,.28,.14], [0,0,.16], [.12,.25,.55])
    for side in [-1,1]:
        h = sim.createPrimitiveShape(sim.primitiveshape_cylinder,[.18,.18,.06],0)
        sim.setObjectInt32Param(h,sim.shapeintparam_static,1)
        sim.setObjectPosition(h,world,[0,side*.18,.1])
        sim.setObjectOrientation(h,world,[math.pi/2,0,0])
        color(h,[.08,.08,.08]); sim.setObjectParent(h,robot,True)
    box('SoporteCamara',[.04,.04,.25],[.14,0,.32],[.2,.2,.2],robot)
    sensor = sim.createVisionSensor(1+2+4, [args.resolution,args.resolution,0,0],
        [.02,10,math.radians(65),.03,0,0,.6,.6,.6,0,0])
    sim.setObjectAlias(sensor,'CamaraFrontal')
    sim.setObjectMatrix(sensor,world,camera_matrix(.2,0,.45,0))
    sim.setObjectParent(sensor,robot,True)
    panel = box('Panel',[.025,.66,.66],[1.5,0,.45],[.92,.92,.92])
    obstacle = box('Obstaculo',[.3,.65,.75],[1.5,0,.375],[.35,.22,.12])
    symbols = {}
    for name, rows in PATTERNS.items():
        vertices, indices = [], []
        for row, cells in enumerate(rows):
            for col, cell in enumerate(cells):
                if cell != '1': continue
                y0, y1 = (3.5-col)*.065, (2.5-col)*.065
                z0, z1 = (3.5-row)*.065, (2.5-row)*.065
                base = len(vertices)//3
                vertices.extend([0,y0,z0,0,y1,z0,0,y1,z1,0,y0,z1])
                indices.extend([base,base+1,base+2,base,base+2,base+3])
        h = sim.createShape(0,0,vertices,indices)
        sim.setObjectAlias(h,'Simbolo_'+name)
        sim.setObjectInt32Param(h,sim.shapeintparam_static,1)
        color(h,[.04,.04,.04]); symbols[name] = h
    def hide(h): sim.setObjectPosition(h,world,[0,0,-8])
    for h in [obstacle,*symbols.values()]: hide(h)
    sim.setObjectPosition(symbols['avanzar'],world,[1.48,0,.45])
    scene = out/'escena_dataset.ttt'
    sim.saveScene(str(scene))
    config = vars(args) | {'classes':CLASSES,'groups_per_class':20,
        'split_groups':{'train':14,'validation':3,'test':3},
        'camera_variation':{'yaw_max':YAW_MAX,'pitch_max':PITCH_MAX,'roll_max':ROLL_MAX,'camera_z':CAM_Z},
        'noise':{'fraction':args.noise_fraction,'sigma':SIGMA,'salt_pepper':PROB_SP},
        'robot':'kinematic acquisition rig; wheels have no motor joints',
        'status':'scene_built' if args.build_only else 'capturing'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    if args.build_only:
        print('Escena guardada:',scene); return
    previews = {}
    total_ruido = 0
    n_ruido = round(args.frames_per_group * args.noise_fraction)   # imágenes con ruido por grupo y clase
    fields = ['file','split','class_id','class_name','group','frame','distance','target_y',
              'camera_yaw','camera_pitch','camera_roll','camera_z','noise','seed']
    try:
        with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for group in range(20):
                split = split_for(group)
                # A scenario group and all its classes stay in the same partition.
                base_distance = rng.uniform(1.15,1.85)
                base_y = rng.uniform(-.12,.12)
                gray = rng.uniform(.28,.65)
                color(floor,[gray]*3)
                color(back,[rng.uniform(.45,.8)]*3)
                for class_id, name in enumerate(CLASSES):
                    folder = out/split/name; folder.mkdir(parents=True,exist_ok=True)
                    for h in [panel,obstacle,*symbols.values()]: hide(h)
                    con_ruido = set(rng.sample(range(args.frames_per_group), n_ruido))
                    for frame in range(args.frames_per_group):
                        distance = base_distance+rng.uniform(-.08,.08)
                        target_y = base_y+rng.uniform(-.025,.025)
                        # Ángulo de cámara variado en cada foto
                        yaw = rng.uniform(-YAW_MAX,YAW_MAX)
                        pitch = rng.uniform(-PITCH_MAX,PITCH_MAX)
                        roll = rng.uniform(-ROLL_MAX,ROLL_MAX)
                        z = rng.uniform(*CAM_Z)
                        sim.setObjectMatrix(sensor,world,camera_matrix(.2,0,z,yaw))
                        # Inclinación y rotación aplicadas en el marco propio del sensor
                        m = sim.getObjectMatrix(sensor,world)
                        giro = sim.buildMatrix([0,0,0],[pitch,0,roll])
                        sim.setObjectMatrix(sensor,world,sim.multiplyMatrices(m,giro))
                        if name == 'obstaculo':
                            sim.setObjectPosition(obstacle,world,[distance,target_y,.375])
                            color(obstacle,[rng.uniform(.15,.55) for _ in range(3)])
                        else:
                            sim.setObjectPosition(panel,world,[distance,target_y,.45])
                            sim.setObjectPosition(symbols[name],world,[distance-.014,target_y,.45])
                        sim.handleVisionSensor(sensor)
                        raw, resolution = sim.getVisionSensorImg(sensor)
                        w,h = resolution
                        array = np.frombuffer(bytes(raw),dtype=np.uint8).reshape(h,w,3)
                        # OpenGL origin is bottom-left. Keep RGB; invert rows only.
                        image = Image.fromarray(np.flipud(array).copy())
                        if np.asarray(image).std() < 2:
                            raise RuntimeError('Imagen casi uniforme: revisa luces y renderizado de cámara.')
                        if args.save_size:
                            image = image.resize((args.save_size,args.save_size),Image.LANCZOS)
                        ruido = frame in con_ruido
                        if ruido:
                            image = agregar_ruido(image)
                            total_ruido += 1
                        relative = Path(split)/name/f'g{group:02d}_f{frame:04d}.png'
                        image.save(out/relative)
                        previews.setdefault((name,ruido),image.copy())
                        writer.writerow(dict(file=relative.as_posix(),split=split,class_id=class_id,
                            class_name=name,group=group,frame=frame,distance=distance,target_y=target_y,
                            camera_yaw=yaw,camera_pitch=pitch,camera_roll=roll,camera_z=z,
                            noise=int(ruido),seed=args.seed))
                    f.flush()
                print(f'Grupo {group+1}/20: {split}',flush=True)
        # Vista previa: fila superior sin ruido, fila inferior con ruido
        cel = 192
        sheet = Image.new('RGB',(len(CLASSES)*cel,2*cel+44),'white'); draw = ImageDraw.Draw(sheet)
        for i,name in enumerate(CLASSES):
            for fila,ruido in enumerate([False,True]):
                if (name,ruido) in previews:
                    sheet.paste(previews[(name,ruido)].resize((cel,cel),Image.NEAREST),(i*cel,fila*(cel+22)))
                draw.text((i*cel+5,fila*(cel+22)+cel+4),name+(' (ruido)' if ruido else ''),fill='black')
        sheet.save(out/'vista_previa.png')
        total = 20*args.frames_per_group*len(CLASSES)
        config['status']='complete'
        config['images']=total
        config['images_with_noise']=total_ruido
    except Exception:
        config['status']='incomplete'
        raise
    finally:
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print(f'Captura completada: {out}')
    print(f'Imágenes con ruido: {total_ruido} de {20*args.frames_per_group*len(CLASSES)}')

if __name__ == '__main__':
    main()