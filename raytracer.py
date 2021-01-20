import numpy as np
import matplotlib.pyplot as plt


class Light:
    """
    Lights have positions and lighty properties
    """
    def __init__(self, centre=[], ambient=[], diffuse=[], specular=[]):
        self.centre = np.asarray(centre)
        self.ambient = np.asarray(ambient)
        self.diffuse = np.asarray(diffuse)
        self.specular = np.asarray(specular)


class Sphere(Light):
    """
    Spheres have all the things lights have,
    as well as radii and reflective properties
    """
    def __init__(self, radius=[], shine=100, reflection=0.5, **kwargs):
        super().__init__(**kwargs)
        self.radius = radius
        self.shine = shine
        self.reflection = reflection


def norm(vec):
    return vec / np.linalg.norm(vec)


def reflected(vec, axis):
    return vec - 2 * np.dot(vec, axis) * axis


def sphere_intersect(obj, ray_origin, ray_direction):
    b = 2 * np.dot(ray_direction, ray_origin - obj.centre)
    c = np.linalg.norm(ray_origin - obj.centre) ** 2 - obj.radius ** 2

    delta = b ** 2 - 4 * c
    if delta > 0:
        t1 = (-b + np.sqrt(delta)) / 2
        t2 = (-b - np.sqrt(delta)) / 2
        if t1 > 0 and t2 > 0:
            return min(t1, t2)

    return np.inf # if they don't intersect, then they intersect at inf


def nearest_intersected_object(objects, ray_origin, ray_direction):
    distances = [sphere_intersect(o, ray_origin, ray_direction) for o in objects]
    near, dist = min(zip(objects, distances), key=lambda x: x[1])
    return near, dist


def render(camera, screen, objects, depth=3):


    light, *objects = objects

    image = np.zeros((screen["height"], screen["width"], 3))

    step_up = np.linspace(screen["top"], screen["bottom"], screen["height"])
    step_across = np.linspace(screen["left"], screen["right"], screen["width"])

    for i, y in enumerate(step_up):
        for j, x in enumerate(step_across):

            pixel = np.array([x, y, 0])

            origin = camera
            direction = norm(pixel - origin)

            color = np.zeros((3))
            reflection = 1

            for _ in range(depth):
                # check for intersections
                near_obj, min_dist = nearest_intersected_object(objects, origin, direction)
                if min_dist == np.inf:
                    break

                intersection = origin + min_dist * direction
                normal_to_surface = norm(intersection - near_obj.centre)

                shifted_point = intersection + 1e-5 * normal_to_surface
                intersection_to_light = norm(light.centre - shifted_point)

                _, min_dist = nearest_intersected_object(objects, shifted_point, intersection_to_light)
                intersection_to_light_distance = np.linalg.norm(light.centre - intersection)

                is_shadowed = min_dist < intersection_to_light_distance
                if is_shadowed:
                    break

                illumination = np.zeros((3))

                # ambiant
                illumination += near_obj.ambient * light.ambient

                # diffuse
                illumination += near_obj.diffuse * light.diffuse * np.dot(intersection_to_light, normal_to_surface)

                # specular
                intersection_to_camera = norm(camera - intersection)

                H = norm(intersection_to_light + intersection_to_camera)
                illumination += near_obj.specular * light.specular * np.dot(normal_to_surface, H) ** (near_obj.shine / 4)

                # reflection
                color += reflection * illumination
                reflection *= near_obj.reflection

                origin = shifted_point
                direction = reflected(direction, normal_to_surface)

            image[i, j] = np.clip(color, 0, 1)

    plt.imsave('image.png', image)


if __name__ == "__main__":

    width = 700
    height = 500

    camera = np.array([0, 0, 1])

    ratio = width / height
    screen = {"left": -1,
              "top": 1 / ratio,
              "right": 1,
              "bottom": -1 / ratio,
              "height": height,
              "width": width}

    light = Light(
                centre   = [5, 5, 5],
                ambient  = [1, 1, 1],
                diffuse  = [1, 1, 1],
                specular = [1, 1, 1]
            )


    objects = [
            light, # light is the first object
            Sphere(
                centre = [-0.2, 0, -1],
                radius = 0.7,
                ambient = [0.1, 0, 0],
                diffuse = [0.7, 0, 0],
                specular = [1, 1, 1]
                ),

            Sphere(
                centre = [0.1, -0.3, 0],
                radius = 0.1,
                ambient = [0.1, 0, 0.1],
                diffuse = [0.7, 0, 0.7],
                specular = [1, 1, 1]
                ),

            Sphere(
                centre = [-0.3, 0, 0],
                radius = 0.15,
                ambient = [0, 0.1, 0],
                diffuse = [0, 0.6, 0],
                specular = [1, 1, 1]
                ),
    ]


    render(camera, screen, objects)
