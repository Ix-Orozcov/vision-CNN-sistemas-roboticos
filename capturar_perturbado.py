"""
Captura un conjunto NUEVO de imágenes perturbadas en CoppeliaSim para verificar la CNN.

Cada imagen aplica UNA de estas perturbaciones (las demás condiciones quedan en su rango nominal):
  - iluminacion      : luces más tenues o más intensas y con tono cálido/frío
  - distancia        : patrón mucho más cerca o más lejos que en el entrenamiento
  - angulo_camara    : giro (yaw), inclinación (pitch), rotación (roll) y altura de la cámara
  - posicion_patron  : patrón desplazado a los lados, arriba/abajo y girado sobre su eje vertical

Las clases y las perturbaciones se reparten de forma equilibrada.
Estructura de salida (la misma que espera la CNN):
    dataset_perturbado/
        avanzar/   000_iluminacion.png, 004_distancia.png, ...
        izquierda/ ...
        derecha/   ...
        obstaculo/ ...
        manifest.csv      (parámetros de cada imagen)
        vista_previa.png  (ejemplo de cada clase x perturbación)

Uso (con CoppeliaSim abierto, escena NUEVA y simulación detenida):
    python capturar_perturbado.py --output dataset_perturbado_moderado --intensidad moderada
    python capturar_perturbado.py --output dataset_perturbado_severo   --intensidad severa
"""
import argparse
import csv
import json
import math
from pathlib import Path
import random

CLASSES = ['avanzar', 'izquierda', 'derecha', 'obstaculo']
PERTURBACIONES = ['iluminacion', 'distancia', 'angulo_camara', 'posicion_patron']

# Symbols are defined as seen by the camera: image right = world -Y.
PATTERNS = {
 'avanzar': ['0001000','0011100','0111110','0001000','0001000','0001000','0001000'],
 'izquierda': ['0001000','0011000','0111111','1111111','0111111','0011000','0001000'],
 'derecha': ['0001000','0001100','1111110','1111111','1111110','0001100','0001000'],
}

TONOS = {'neutro': [1.0, 1.0, 1.0], 'calido': [1.0, .82, .6], 'frio': [.65, .82, 1.0]}

# ---------------- Intensidad de las perturbaciones ----------------
# 'severa'   : los rangos originales (lejos del rango de entrenamiento)
# 'moderada' : variaciones realistas, justo fuera del rango de entrenamiento
RANGOS = {
    'severa': dict(
        luz_baja=(.25, .60), luz_alta=(1.5, 2.2), fuerza_tono=1.0, var_ambiente=(.7, 1.3),
        dist_cerca=(.80, 1.05), dist_lejos=(2.0, 2.6),
        yaw=(.10, .20), pitch=(.06, .15), roll=(.08, .25), cam_z=(.32, .62),
        despl_y=(.18, .36), altura_patron=(.33, .60), giro_panel=(.25, .50),
    ),
    'moderada': dict(
        luz_baja=(.50, .75), luz_alta=(1.25, 1.6), fuerza_tono=0.5, var_ambiente=(.85, 1.15),
        dist_cerca=(.95, 1.10), dist_lejos=(1.9, 2.2),
        yaw=(.06, .12), pitch=(.04, .09), roll=(.05, .14), cam_z=(.38, .54),
        despl_y=(.14, .24), altura_patron=(.38, .52), giro_panel=(.12, .28),
    ),
}


def camera_matrix(x, y, z, yaw):
    # Columns: sensor right, sensor up, sensor forward. Forward is local +Z.
    c, s = math.cos(yaw), math.sin(yaw)
    return [s,0,c,x, -c,0,s,y, 0,1,0,z]


def signo(rng):
    return rng.choice([-1, 1])


def parametros_nominales(rng):
    """Mismos rangos que se usaron para capturar train/validation/test."""
    return dict(
        distance=rng.uniform(1.15, 1.85),
        target_y=rng.uniform(-.12, .12),
        target_z=.45,
        panel_yaw=0.0,
        camera_yaw=rng.uniform(-.035, .035),
        camera_pitch=0.0,
        camera_roll=0.0,
        camera_z=rng.uniform(.43, .47),
        light_factor=1.0,
        ambient_factor=1.0,
        light_tint='neutro',
    )


def aplicar_perturbacion(tipo, p, rng, r):
    """Modifica SOLO los parámetros de la perturbación indicada, usando los rangos r."""
    if tipo == 'iluminacion':
        if rng.random() < .5:
            p['light_factor'] = rng.uniform(*r['luz_baja'])   # escena oscura
        else:
            p['light_factor'] = rng.uniform(*r['luz_alta'])  # escena muy iluminada
        p['ambient_factor'] = p['light_factor'] * rng.uniform(*r['var_ambiente'])
        p['light_tint'] = rng.choice(list(TONOS))
    elif tipo == 'distancia':
        if rng.random() < .5:
            p['distance'] = rng.uniform(*r['dist_cerca'])     # más cerca que en entrenamiento
        else:
            p['distance'] = rng.uniform(*r['dist_lejos'])     # más lejos que en entrenamiento
    elif tipo == 'angulo_camara':
        p['camera_yaw'] = signo(rng) * rng.uniform(*r['yaw'])
        p['camera_pitch'] = signo(rng) * rng.uniform(*r['pitch'])
        p['camera_roll'] = signo(rng) * rng.uniform(*r['roll'])
        p['camera_z'] = rng.uniform(*r['cam_z'])
    elif tipo == 'posicion_patron':
        p['target_y'] = signo(rng) * rng.uniform(*r['despl_y'])     # desplazado a un lado
        p['target_z'] = rng.uniform(*r['altura_patron'])            # más arriba o más abajo
        p['panel_yaw'] = signo(rng) * rng.uniform(*r['giro_panel']) # panel girado
    return p


def main():
    p = argparse.ArgumentParser(description='Captura imágenes perturbadas para verificar la CNN.')
    p.add_argument('--output', default='dataset_perturbado')
    p.add_argument('--host', default='localhost')
    p.add_argument('--port', type=int, default=23000)
    p.add_argument('--seed', type=int, default=2026)
    p.add_argument('--total', type=int, default=75)
    p.add_argument('--resolution', type=int, default=320)
    p.add_argument('--intensidad', choices=list(RANGOS), default='moderada',
                   help="'moderada' (realista) o 'severa' (rangos originales).")
    args = p.parse_args()
    rangos = RANGOS[args.intensidad]
    if args.total < 1 or args.resolution < 64:
        p.error('Usa al menos 1 imagen y resolución >=64.')
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
    world = sim.handle_world

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

    # ---------------- Escena (igual que el script original) ----------------
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
    symbols, symbol_pose = {}, {}
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
        # Pose del símbolo relativa al panel (para que lo siga si el panel se mueve o gira)
        sim.setObjectPosition(h,world,[1.5-.014,0,.45])
        symbol_pose[name] = (sim.getObjectPosition(h,panel), sim.getObjectOrientation(h,panel))

    def hide(h): sim.setObjectPosition(h,world,[0,0,-8])
    for h in [obstacle,*symbols.values()]: hide(h)
    sim.setObjectPosition(symbols['avanzar'],world,[1.48,0,.45])
    scene = out/'escena_perturbada.ttt'
    sim.saveScene(str(scene))

    # ---------------- Control de iluminación ----------------
    luces = {}
    for h in sim.getObjectsInTree(sim.handle_scene, sim.object_light_type, 0):
        try:
            state, _, diffuse, specular = sim.getLightParameters(h)
            luces[h] = (state, list(diffuse), list(specular))
        except Exception:
            pass
    try:
        ambiente = list(sim.getArrayParam(sim.arrayparam_ambient_light))
    except Exception:
        ambiente = None

    def set_iluminacion(factor, ambient_factor, tint):
        """Escala las luces de la escena. Devuelve True si el simulador aceptó el cambio."""
        ok = False
        for h, (state, diffuse, specular) in luces.items():
            try:
                sim.setLightParameters(h, state, None,
                    [min(1.0, d*factor*t) for d, t in zip(diffuse, tint)],
                    [min(1.0, s*factor) for s in specular])
                ok = True
            except Exception:
                pass
        if ambiente is not None:
            try:
                sim.setArrayParam(sim.arrayparam_ambient_light,
                    [min(1.0, a*ambient_factor*t) for a, t in zip(ambiente, tint)])
                ok = True
            except Exception:
                pass
        return ok

    def iluminacion_posproceso(array, factor, tint):
        """Respaldo si la API de luces no está disponible: simula la luz sobre la imagen
        (ganancia global + tono + degradado como de una luz direccional)."""
        h, w, _ = array.shape
        ang = rng.uniform(0, 2*math.pi)
        X, Y = np.meshgrid(np.linspace(-1,1,w), np.linspace(-1,1,h))
        degradado = 1 + .35*(math.cos(ang)*X + math.sin(ang)*Y)
        salida = array.astype(np.float32)*factor*degradado[..., None]*np.array(tint, dtype=np.float32)
        return np.clip(salida, 0, 255).astype(np.uint8)

    # ---------------- Plan de captura (equilibrado) ----------------
    # Cuadrado latino: cada bloque de 16 imágenes contiene cada (clase, perturbación) una vez.
    plan = [(CLASSES[k % 4], PERTURBACIONES[(k + k//4) % 4]) for k in range(args.total)]

    config = vars(args) | {'classes':CLASSES, 'perturbaciones':PERTURBACIONES, 'status':'capturing'}
    (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    fields = ['file','class_id','class_name','perturbacion','distance','target_y','target_z','panel_yaw',
              'camera_yaw','camera_pitch','camera_roll','camera_z','light_factor','ambient_factor',
              'light_tint','light_mode','seed']
    previews = {}
    aviso_luz = False
    try:
        with (out/'manifest.csv').open('w',newline='',encoding='utf-8') as f:
            writer = csv.DictWriter(f,fieldnames=fields); writer.writeheader()
            for k, (name, tipo) in enumerate(plan):
                prm = aplicar_perturbacion(tipo, parametros_nominales(rng), rng, rangos)
                folder = out/name; folder.mkdir(parents=True,exist_ok=True)

                # Fondo y suelo con la misma variación nominal que el dataset original
                color(floor,[rng.uniform(.28,.65)]*3)
                color(back,[rng.uniform(.45,.8)]*3)

                # Iluminación (nominal en todas las imágenes salvo las de 'iluminacion')
                # El tono se suaviza según la intensidad (0.5 = mitad de fuerte)
                tint = [1 - rangos['fuerza_tono']*(1 - t) for t in TONOS[prm['light_tint']]]
                light_mode = 'nominal'
                if tipo == 'iluminacion':
                    if set_iluminacion(prm['light_factor'], prm['ambient_factor'], tint):
                        light_mode = 'simulador'
                    else:
                        light_mode = 'posproceso'
                        if not aviso_luz:
                            print('Aviso: no se pudieron cambiar las luces de CoppeliaSim; '
                                  'la iluminación se simulará sobre la imagen.')
                            aviso_luz = True
                else:
                    set_iluminacion(1.0, 1.0, TONOS['neutro'])

                # Cámara: yaw con la matriz original; pitch y roll como giros en el marco del sensor
                sim.setObjectMatrix(sensor,world,camera_matrix(.2,0,prm['camera_z'],prm['camera_yaw']))
                if prm['camera_pitch'] or prm['camera_roll']:
                    m = sim.getObjectMatrix(sensor,world)
                    giro = sim.buildMatrix([0,0,0],[prm['camera_pitch'],0,prm['camera_roll']])
                    sim.setObjectMatrix(sensor,world,sim.multiplyMatrices(m,giro))

                # Objetivo
                for h in [panel,obstacle,*symbols.values()]: hide(h)
                d, y, z, ang = prm['distance'], prm['target_y'], prm['target_z'], prm['panel_yaw']
                if name == 'obstaculo':
                    sim.setObjectPosition(obstacle,world,[d,y,.375])
                    sim.setObjectOrientation(obstacle,world,[0,0,ang])
                    color(obstacle,[rng.uniform(.15,.55) for _ in range(3)])
                else:
                    sim.setObjectPosition(panel,world,[d,y,z])
                    sim.setObjectOrientation(panel,world,[0,0,ang])
                    rel_pos, rel_ori = symbol_pose[name]
                    sim.setObjectPosition(symbols[name],panel,rel_pos)
                    sim.setObjectOrientation(symbols[name],panel,rel_ori)

                # Captura
                sim.handleVisionSensor(sensor)
                raw, resolution = sim.getVisionSensorImg(sensor)
                w,h = resolution
                array = np.frombuffer(bytes(raw),dtype=np.uint8).reshape(h,w,3)
                # OpenGL origin is bottom-left. Keep RGB; invert rows only.
                array = np.flipud(array).copy()
                if light_mode == 'posproceso':
                    array = iluminacion_posproceso(array, prm['light_factor'], tint)
                image = Image.fromarray(array)
                if np.asarray(image).std() < 2:
                    raise RuntimeError('Imagen casi uniforme: revisa luces y renderizado de cámara.')
                relative = Path(name)/f'{k:03d}_{tipo}.png'
                image.save(out/relative)
                previews.setdefault((name,tipo),image.copy())
                writer.writerow({**{c: prm[c] for c in fields if c in prm},
                                 'file':relative.as_posix(),'class_id':CLASSES.index(name),
                                 'class_name':name,'perturbacion':tipo,
                                 'light_mode':light_mode,'seed':args.seed})
                f.flush()
                print(f'{k+1}/{args.total}: {name:<10} | {tipo}',flush=True)

        # Vista previa: filas = clases, columnas = perturbaciones
        cel = 160
        sheet = Image.new('RGB',(4*cel+90,4*cel+24),'white'); draw = ImageDraw.Draw(sheet)
        for j,tipo in enumerate(PERTURBACIONES):
            draw.text((90+j*cel+5,6),tipo,fill='black')
        for i,name in enumerate(CLASSES):
            draw.text((5,24+i*cel+cel//2),name,fill='black')
            for j,tipo in enumerate(PERTURBACIONES):
                if (name,tipo) in previews:
                    sheet.paste(previews[(name,tipo)].resize((cel,cel)),(90+j*cel,24+i*cel))
        sheet.save(out/'vista_previa.png')
        config['status']='complete'
        config['images']=args.total
    except Exception:
        config['status']='incomplete'
        raise
    finally:
        set_iluminacion(1.0, 1.0, TONOS['neutro'])   # dejar las luces como estaban
        (out/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    print('Captura completada:',out)


if __name__ == '__main__':
    main()