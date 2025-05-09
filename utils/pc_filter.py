import numpy as np
import os
import argparse

def filter_point_cloud_by_distance(point_cloud, max_distance):
    """
    Filters a point cloud based on the distance from the origin.

    Parameters:
        point_cloud (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud,
                                      where N is the number of points.
        max_distance (float): The maximum distance from the origin to keep a point.

    Returns:
        numpy.ndarray: A filtered numpy array containing only the points within the specified distance.
    """
    if point_cloud.shape[1] != 3:
        raise ValueError("Point cloud must have shape (N, 3)")

    distances = np.linalg.norm(point_cloud, axis=1)
    filtered_points = point_cloud[distances <= max_distance]
    return filtered_points

def filter_point_cloud_by_motion(point_cloud, motion_vectors, max_motion):
    """
    Filters a point cloud based on the magnitude of motion vectors.

    Parameters:
        point_cloud (numpy.ndarray): A numpy array of shape (N, 3) representing the point cloud,
                                        where N is the number of points.
        motion_vectors (numpy.ndarray): A numpy array of shape (N, 3) representing the motion vectors
                                            corresponding to each point in the point cloud.
        max_motion (float): The maximum motion magnitude to keep a point.

    Returns:
        numpy.ndarray: A filtered numpy array containing only the points with motion magnitude
                        less than or equal to the specified maximum motion.
    """
    if point_cloud.shape[1] != 3 or motion_vectors.shape[1] != 3:
        raise ValueError("Point cloud and motion vectors must have shape (N, 3)")
    if point_cloud.shape[0] != motion_vectors.shape[0]:
        raise ValueError("Point cloud and motion vectors must have the same number of points")

    motion_magnitudes = np.linalg.norm(motion_vectors, axis=1)
    filtered_points = point_cloud[motion_magnitudes <= max_motion]
    return filtered_points


def combine_npy_files(input_folder, output_file):
    """
    Combines all .npy files in a folder into a single .npy file and saves it.

    Parameters:
        input_folder (str): Path to the folder containing .npy files.
        output_file (str): Path to the output .npy file.
    """
    combined_data = []
    for file_name in os.listdir(input_folder):
        if file_name.endswith('.npy'):
            file_path = os.path.join(input_folder, file_name)
            data = np.load(file_path)
            combined_data.append(data)
    
    combined_data = np.concatenate(combined_data, axis=1)
    np.save(output_file, combined_data)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Combine .npy files in a directory into a single .npy file.")
    parser.add_argument("--input_dir", type=str, help="Path to the folder containing .npy files.")
    parser.add_argument("--output", type=str, help="Path to the output .npy file.")
    args = parser.parse_args()

    combine_npy_files(args.input_dir, args.output)