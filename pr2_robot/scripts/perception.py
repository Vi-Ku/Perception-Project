#!/usr/bin/env python

# Import modules
import numpy as np
import sklearn
from sklearn.preprocessing import LabelEncoder
import pickle
from sensor_stick.srv import GetNormals
from sensor_stick.features import compute_color_histograms
from sensor_stick.features import compute_normal_histograms
from visualization_msgs.msg import Marker
from sensor_stick.marker_tools import *
from sensor_stick.msg import DetectedObjectsArray
from sensor_stick.msg import DetectedObject
from sensor_stick.pcl_helper import *

import rospy
import tf
from geometry_msgs.msg import Pose
from std_msgs.msg import Float64
from std_msgs.msg import Int32
from std_msgs.msg import String
from pr2_robot.srv import *
from rospy_message_converter import message_converter
import yaml


# Helper function to get surface normals
'''
Calculates the surface normal of each point in the point cloud
:param: cloud: point cloud to get the point normals
:return: array of point surface normals
'''
def get_normals(cloud):
    get_normals_prox = rospy.ServiceProxy('/feature_extractor/get_normals', GetNormals)
    return get_normals_prox(cloud).cluster

# Helper function to create a yaml friendly dictionary from ROS messages
'''
Converts the supplied data into a yaml friendly dictionary from ROS messages dictionary
:param test_scene_num: enviroment scene used with the robot
:param arm_name: arm used to pickup the object
:param object_name: the objectes label that is to be picked up
:param pick_pose: position of the object
:param place_pose: position of the objects final destination
:return: data dictionary
'''
def make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose):
    yaml_dict = {}
    yaml_dict["test_scene_num"] = test_scene_num.data
    yaml_dict["arm_name"]  = arm_name.data
    yaml_dict["object_name"] = object_name.data
    yaml_dict["pick_pose"] = message_converter.convert_ros_message_to_dictionary(pick_pose)
    yaml_dict["place_pose"] = message_converter.convert_ros_message_to_dictionary(place_pose)
    return yaml_dict

# Helper function to output to yaml file
'''
saves data output to yaml file
:param yaml_filename: the name of the file to be saved
:param dict_list: the data to be saved
:return: None
'''
def send_to_yaml(yaml_filename, dict_list):
    data_dict = {"object_list": dict_list}
    with open(yaml_filename, 'w') as outfile:
        yaml.dump(data_dict, outfile, default_flow_style=False)

# Callback function for your Point Cloud Subscriber
def pcl_callback(pcl_msg):

# Exercise-2 TODOs:

    # TODO: Convert ROS msg to PCL data
    cloud = ros_to_pcl(pcl_msg)

    # TODO: Statistical Outlier Filtering
    cloud = XYZRGB_to_XYZ(cloud)
    outlier_filter = cloud.make_statistical_outlier_filter()
        # Set the number of neighboring points to analyze for any given point
    outlier_filter.set_mean_k(5)
        # Set threshold scale factor
    x = 0.0005
        # Any point with a mean distance larger than global (mean distance+x*std_dev) will be considered outlier
    outlier_filter.set_std_dev_mul_thresh(x)
        # Finally call the filter function for magic
    outliers_removed = outlier_filter.filter()
    outliers_removed = XYZ_to_XYZRGB(outliers_removed,[255,255,255])

    #----------------------------------------------------#
    pcl_static_out = pcl_to_ros(outliers_removed)
    pcl_static_out_pub.publish(pcl_static_out)
    #----------------------------------------------------#

    # TODO: Voxel Grid Downsampling
        # Create a VoxelGrid filter object for our input point cloud
    vox = outliers_removed.make_voxel_grid_filter()
    LEAF_SIZE = 0.005
        # Set the voxel (or leaf) size  
    vox.set_leaf_size(LEAF_SIZE, LEAF_SIZE, LEAF_SIZE)
        # Call the filter function to obtain the resultant downsampled point cloud
    cloud_filtered = vox.filter()

    #----------------------------------------------------#
    pcl_vox = pcl_to_ros(cloud_filtered)
    pcl_vox_pub.publish(pcl_vox)
    #----------------------------------------------------#


    # TODO: PassThrough Filter
        # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
        # Assign axis and range to the passthrough filter object.
    filter_axis = 'z'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = 0.6
    axis_max = 1.1
    passthrough.set_filter_limits(axis_min, axis_max)
        # Finally use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()

    #----------------------------------------------------------------------------------
    # PassThrough filter (Y Axis)
    #----------------------------------------------------------------------------------
    # Create a PassThrough filter object.
    passthrough = cloud_filtered.make_passthrough_filter()
    # Assign axis and range to the passthrough filter object.
    filter_axis = 'y'
    passthrough.set_filter_field_name(filter_axis)
    axis_min = -0.456
    axis_max =  0.456
    passthrough.set_filter_limits(axis_min, axis_max)
    # Use the filter function to obtain the resultant point cloud. 
    cloud_filtered = passthrough.filter()


    #----------------------------------------------------#
    pcl_pass_through = pcl_to_ros(cloud_filtered)
    pcl_pass_through_pub.publish(pcl_pass_through)
    #----------------------------------------------------#

    # TODO: RANSAC Plane Segmentation
        # Create the segmentation object
    seg = cloud_filtered.make_segmenter()
        # Set the model you wish to fit 
    seg.set_model_type(pcl.SACMODEL_PLANE)
    seg.set_method_type(pcl.SAC_RANSAC)
        # Max distance for a point to be considered fitting the model
        # Experiment with different values for max_distance 
        # for segmenting the table
    max_distance = 0.006
    seg.set_distance_threshold(max_distance)
        # Call the segment function to obtain set of inlier indices and model coefficients
    inliers, coefficients = seg.segment()

    # TODO: Extract inliers and outliers
    cloud_table = cloud_filtered.extract(inliers, negative=False)
        # Extract outliers
    cloud_objects = cloud_filtered.extract(inliers, negative=True)

    # TODO: Euclidean Clustering
    white_cloud = XYZRGB_to_XYZ(cloud_objects)
    tree = white_cloud.make_kdtree()
        # Create a cluster extraction object
    ec = white_cloud.make_EuclideanClusterExtraction()
        # Set tolerances for distance threshold 
        # as well as minimum and maximum cluster size (in points)
        # NOTE: These are poor choices of clustering parameters
        # Your task is to experiment and find values that work for segmenting objects.
    ec.set_ClusterTolerance(0.03)
    ec.set_MinClusterSize(30)
    ec.set_MaxClusterSize(5000)
        # Search the k-d tree for clusters
    ec.set_SearchMethod(tree)
        # Extract indices for each of the discovered clusters
    cluster_indices = ec.Extract()
        #Assign a color corresponding to each segmented object in scene
    cluster_color = get_color_list(len(cluster_indices))
    color_cluster_point_list = []

    for j, indices in enumerate(cluster_indices):
        for i, indice in enumerate(indices):
            color_cluster_point_list.append([white_cloud[indice][0],
                                            white_cloud[indice][1],
                                            white_cloud[indice][2],
                                            rgb_to_float(cluster_color[j])])

        #Create new cloud containing all clusters, each with unique color
    cluster_cloud = pcl.PointCloud_PointXYZRGB()
    cluster_cloud.from_list(color_cluster_point_list)

    # TODO: Convert PCL data to ROS messages
    pcl_msg_object = pcl_to_ros(cloud_objects)
    pcl_msg_table = pcl_to_ros(cloud_table)
    pcl_msg_cluster = pcl_to_ros(cluster_cloud)

    # # TODO: Publish ROS messages
    pcl_objects_pub.publish(pcl_msg_object)
    pcl_table_pub.publish(pcl_msg_table)
    pcl_cluster_pub.publish(pcl_msg_cluster)


    # Publish the list of detected objects
    detected_objects_labels = []
    detected_objects = []


    for index, pts_list in enumerate(cluster_indices):
        pcl_cluster = cloud_objects.extract(pts_list)

        # Convert the cluster from pcl to ROS using helper function
        ros_cluster = pcl_to_ros(pcl_cluster)
        
        # Extract histogram features
        chists = compute_color_histograms(ros_cluster, using_hsv=False)
        normals = get_normals(ros_cluster)
        nhists = compute_normal_histograms(normals)
        feature = np.concatenate((chists, nhists))
        
        # Make the prediction, retrieve the label for the result
        # and add it to detected_objects_labels list
        prediction = clf.predict(scaler.transform(feature.reshape(1,-1)))
        label = encoder.inverse_transform(prediction)[0]
        detected_objects_labels.append(label)

        # Publish a label into RViz
        label_pos = list(white_cloud[pts_list[0]])
        label_pos[2] += .4
        object_markers_pub.publish(make_label(label,label_pos, index))

        # Add the detected object to the list of detected objects.
        do = DetectedObject()
        do.label = label
        do.cloud = ros_cluster
        detected_objects.append(do)

    rospy.loginfo('Detected {} objects: {}'.format(len(detected_objects_labels), detected_objects_labels))

    # Publish the list of detected objects
    # This is the output you'll need to complete the upcoming project!
    detected_objects_pub.publish(detected_objects)

    # Suggested location for where to invoke your pr2_mover() function within pcl_callback()
    # Could add some logic to determine whether or not your object detections are robust
    # before calling pr2_mover()

    if len(detected_objects)>0:
        try:
            pr2_mover(detected_objects)
        except rospy.ROSInterruptException:
            pass
    else:
        rospy.logwarn('No detected objects !!!')
    
    return 

# function to load parameters and request PickPlace service
def pr2_mover(object_list):
    
    dict_list = []

    # Get/Read the detected object parameters
    object_list_param = rospy.get_param('/object_list')

    # Loop through the pick list
    labels = []
    centroids = []  # to be list of tuples (x, y, z)

    for i in range(len(object_list_param)):
        # Put the object parameters into individual variables
        object_name = object_list_param[i]['name']
        object_group = object_list_param[i]['group']

        # TODO: Rotate PR2 in place to capture side tables for the collision map
        #topic_name = "/pr2/world_joint_controller/command"
        #pub_PR2_rotate = rospy.Publisher(topic_name, std_msgs.msg.Float64, queue_size=3)
        #pub_PR2_rotate.publish(np.pi/2)

        for j, object in enumerate(object_list):
            labels.append(object.label)
            points_arr = ros_to_pcl(object.cloud).to_array()
            centroids.append(np.mean(points_arr, axis=0)[:3])

            # check if the object is the next object on the list to relocate
            if (object.label == object_name):
                labels.append(object.label)
                points_arr = ros_to_pcl(object.cloud).to_array()

                # Get the PointCloud for a given object and obtain it's centroid
                centroids.append(np.mean(points_arr, axis=0)[:3])
                center_x = np.asscalar(centroids[j][0])
                center_y = np.asscalar(centroids[j][1])
                center_z = np.asscalar(centroids[j][2])
                
                # store which environment is used
                test_scene_num = Int32()
                test_scene_num.data =

                # Initialize a variable
                object_name = String()
                # Populate the data field
                object_name.data = object_list_param[i]['name']

                # Create the objects position to be picked up from
                # initialize an empty pose message
                pick_pose = Pose()
                pick_pose.position.x = center_x
                pick_pose.position.y = center_y
                pick_pose.position.z = center_z

                cent = (center_x, center_y, center_z)

                # get parameters
                dropbox_param = rospy.get_param('/dropbox')
                place_pose = Pose()

                # check which box the object is to be dropped into and assign that boxes arm
                arm_name = String()
                if(object_group == dropbox_param[0]['group']):
                    # Assign the left arm to be used for pick_place
                    arm_name.data = dropbox_param[0]['name']

                    # Create the objects final resting position to be placed
                    # offset each item so they do not stack on top of each other
                    # TODO: better object drop location tracking to prevent objects from dropping outside the boxes with large picking list.
                    place_pose.position.x = dropbox_param[0]['position'][0]-0.03 - 0.05 * i
                    place_pose.position.y = dropbox_param[0]['position'][1] + 0.03 * i
                    place_pose.position.z = dropbox_param[0]['position'][2]
                elif(object_group == dropbox_param[1]['group']):
                    # Assign the right arm to be used for pick_place
                    arm_name.data = dropbox_param[1]['name']

                    # Create the objects final resting position to be placed
                    # TODO: As described in the above TODO. better drop location tracking
                    place_pose.position.x = dropbox_param[1]['position'][0]-0.03 - 0.05 * i
                    place_pose.position.y = dropbox_param[1]['position'][1] - 0.06 * i
                    place_pose.position.z = dropbox_param[1]['position'][2]

                # make a dictionary of the robot and objects movement data
                yaml_dict = {}
                yaml_dict = make_yaml_dict(test_scene_num, arm_name, object_name, pick_pose, place_pose)
                dict_list.append(yaml_dict)

                # Wait for 'pick_place_routine' service to come up
                rospy.wait_for_service('pick_place_routine')

                try:
                    pick_place_routine = rospy.ServiceProxy('pick_place_routine', PickPlace)

                    # send a service request for the objects pick and place movement
                    # pick_pose - the position of the object location in which it will be picked up
                    # place_pose - the location of the final place it is to be dropped at
                    resp = pick_place_routine(test_scene_num, object_name, arm_name, pick_pose, place_pose)

                    print ("Response: ",resp.success)

                except rospy.ServiceException, e:
                    print "Service call failed: %s"%e


    # Output the request parameters into the enviroment output yaml file
    out_file = ''
    if (scene_num == 1): out_file = 'output_1.yaml'
    if (scene_num == 2): out_file = 'output_2.yaml'
    if (scene_num == 3): out_file = 'output_3.yaml'
    send_to_yaml(out_file, dict_list)



if __name__ == '__main__':

    rospy.init_node('clustering', anonymous=True)

    # TODO: Create Subscribers
    pcl_sub = rospy.Subscriber("/pr2/world/points", pc2.PointCloud2, pcl_callback, queue_size=1)
    
    # Publishers to visualise the output of static outlier filter, vox filter, pass_through
    # RANSAC filter, segmentation 

    pcl_static_out_pub = rospy.Publisher("/pcl_static_out", PointCloud2, queue_size=1)
    pcl_vox_pub = rospy.Publisher("/pcl_vox", PointCloud2, queue_size=1)
    pcl_pass_through_pub = rospy.Publisher("/pcl_pass_through", PointCloud2, queue_size=1)
    pcl_objects_pub = rospy.Publisher("/pcl_objects", PointCloud2, queue_size=1)
    pcl_table_pub = rospy.Publisher("/pcl_table", PointCloud2, queue_size=1)
    pcl_cluster_pub = rospy.Publisher("/pcl_cluster", PointCloud2, queue_size=1)

    # Publisher for detected object with markers,,,,,,,,,,,,,,,,,
    object_markers_pub = rospy.Publisher("/object_markers", Marker, queue_size=100)
    detected_objects_pub = rospy.Publisher("/detected_objects", DetectedObjectsArray, queue_size=1)
    # pr2_base_mover_pub   = rospy.Publisher("/pr2/world_joint_controller/command", Float64, queue_size=10)

    # Initialize color_list
    get_color_list.color_list = []

    # TODO: Load Model From disk
    model = pickle.load(open('model.sav', 'rb'))
    clf = model['classifier']
    encoder = LabelEncoder()
    encoder.classes_ = model['classes']
    scaler = model['scaler'] 

    # TODO: Spin while node is not shutdown
    while not rospy.is_shutdown():
        rospy.spin()