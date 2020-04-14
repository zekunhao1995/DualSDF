import numpy as np

def calc_area(vertex):
    vec_a = vertex[:,1] - vertex[:,0]
    vec_b = vertex[:,2] - vertex[:,0]
    normal = np.cross(vec_a, vec_b)
    area = np.absolute(np.linalg.norm(normal, ord=2, axis=1))*0.5
    return area

def uniform_sample_on_triangle(triangle):
    while True:
        rn = np.random.rand(2)
        if np.sum(rn) <= 1.0:
            break
    return rn[0]*(triangle[1]-triangle[0]) + rn[1]*(triangle[2]-triangle[0]) + triangle[0]

# mesh
def mesh2pcl(triangle_collection, numpoints):
    area_collection = calc_area(triangle_collection)
    total_area = np.sum(area_collection)
    
    print("Triangle count: {}".format(triangle_collection.shape[0]))
    #print("Total surface area: {}".format(total_area))
    
    area_collection /= total_area
    
    # sample k points
    # note that this will give an error if self.area_collection.shape[0] = 0 (implies empty shape)
    sampled_triangles = np.random.choice(area_collection.shape[0], size=numpoints, p=area_collection)
    
    # Sample one random uvs on each triangle
    rand_uv = np.random.rand(numpoints, 2)
    oob_idx = np.sum(rand_uv, axis=-1) > 1.0
    rand_uv[oob_idx,:] = -rand_uv[oob_idx,:] + 1.0
    
    sampled_triangle_collection = triangle_collection[sampled_triangles,:,:]
    sampled_points =  rand_uv[:,[0]] * (sampled_triangle_collection[:,1,:] - sampled_triangle_collection[:,0,:]) \
                    + rand_uv[:,[1]] * (sampled_triangle_collection[:,2,:] - sampled_triangle_collection[:,0,:]) \
                    + sampled_triangle_collection[:,0,:]
    
    return sampled_points.astype(np.float32)

