#include<opencv2/opencv.hpp>
#include "utilities.h"

using namespace std;
using namespace cv;

ObjectMatching::ObjectMatching(std::vector< cv::Mat > test_images, std::vector< cv::Mat > masks, std::vector< cv::Mat > templates, int obj )
{
	img_tests = test_images;
	img_masks = masks;
	img_template = templates;
	object = obj;
}
/**
	Function to load the test images, the images of the templates and the images of the templates masks
	and to save them in three different vectors.
*/
void ObjectMatching::load_information(cv::String path)
{
	vector<String> test_img_names;
	String pattern = "*.jpg";
	String complete_path = path + "/test_images";
	utils::fs::glob(complete_path, pattern, test_img_names);

	if (test_img_names.size() == 0)
	{
		cout << "None test images " << endl;
		exit(0);
	}

	int i;
	//Save the test images
	for (i = 0; i < test_img_names.size(); i++)
		img_tests.push_back(imread(test_img_names[i]));

	pattern = "*.png";
	complete_path = path + "/models";
	vector<String> models_img_names;
	utils::fs::glob(complete_path, pattern, models_img_names);

	if (models_img_names.size() == 0)
	{
		cout << "None models images" << endl;
		exit(0);
	}

	//Save the masks
	for (i = 0; i < 250; i++)
		img_masks.push_back(imread(models_img_names[i]));

	//Save the template images
	for (; i <models_img_names.size(); i++)
		img_template.push_back(imread(models_img_names[i]));

}

/**
	Function to detect the edges of the test images and the template images
	and save the results in two vectors.
	The thresholds for the Canny function applied to the test images and the template images are differents and 
	given as input parameters.
*/

void ObjectMatching::edge_detection(vector<Mat> tests, vector<Mat> templates, int test_threshold1, int test_threshold2, int template_threshold1, int template_threshold2)
{
	cout << "Detecting the edges ... " << endl;

	//Detect the edges from the test images
	int i;
	for (i = 0; i < tests.size(); i++)
	{
		Mat bw = tests[i];
		Mat edges;
		Canny(bw, edges, test_threshold1, test_threshold2, 3);
		edges_tests.push_back(edges);
	}

	//Detect the edges from the template images
	for (i = 0; i < img_template.size(); i++)
	{
		Mat bw, edges;
		cvtColor(templates[i], bw, COLOR_BGR2GRAY);
		Canny(bw, edges, template_threshold1, template_threshold2, 3);
		edges_templates.push_back(edges);
	}
}


/**
	Function to compute the template matching between the test images and the template images.
	Save the best 60 matches in a vector.
*/

void ObjectMatching::template_matching()
{
	cout << "Computing template matching ... " << endl;
	int i;
	for (i = 0; i < edges_tests.size(); i++)
	{
		vector<Point> vec_pnt;
		vector<int> vec_int;
		vector<double> vec_double;
		best_sixty_locations.push_back(vec_pnt);
		best_sixty_templates.push_back(vec_int);
		best_sixty_values.push_back(vec_double);

		int j;
		Mat result;
		vector<int> best_matches;
		vector<double> best_values;
		vector<Point> positions;
		Point max_location;
		double max_val;
		for (j = 0; j < edges_templates.size(); j++)
		{
			matchTemplate(edges_tests[i], edges_templates[j], result, TM_CCORR_NORMED);
			minMaxLoc(result, NULL, &max_val, NULL, &max_location);
			best_values.push_back(max_val);
			positions.push_back(max_location);
		}

		size = 60;

		for (j = 0; j < best_values.size(); j++)
		{
			if (j < size)
			{
				best_sixty_locations[i].push_back(positions[j]);
				best_sixty_templates[i].push_back(j);
				best_sixty_values[i].push_back(best_values[j]);

				int k = j;
				while (k>0 && best_sixty_values[i][k] > best_sixty_values[i][k-1])
				{
					double temp1 = best_sixty_values[i][k - 1];
					best_sixty_values[i][k - 1] = best_sixty_values[i][k];
					best_sixty_values[i][k] = temp1;

					int temp2 = best_sixty_templates[i][k - 1];
					best_sixty_templates[i][k - 1] = best_sixty_templates[i][k];
					best_sixty_templates[i][k] = temp2;

					Point temp3 = best_sixty_locations[i][k - 1];
					best_sixty_locations[i][k - 1] = best_sixty_locations[i][k];
					best_sixty_locations[i][k] = temp3;

					k--;
				}
			}
			else if (best_values[j] > best_sixty_values[i][size - 1])
			{
				best_sixty_values[i][size -1] = best_values[j];
				best_sixty_templates[i][size -1] = j;
				best_sixty_locations[i][size -1] = positions[j];

				int k = size -1;
				while (k > 0 && best_sixty_values[i][k] > best_sixty_values[i][k - 1] )
				{
					double temp1 = best_sixty_values[i][k - 1];
					best_sixty_values[i][k - 1] = best_sixty_values[i][k];
					best_sixty_values[i][k] = temp1;

					int temp2 = best_sixty_templates[i][k - 1];
					best_sixty_templates[i][k - 1] = best_sixty_templates[i][k];
					best_sixty_templates[i][k] = temp2;

					Point temp3 = best_sixty_locations[i][k - 1];
					best_sixty_locations[i][k - 1] = best_sixty_locations[i][k];
					best_sixty_locations[i][k] = temp3;

					k--;
				}
			}
		}
	}
}

/**
	Function to highlight the contrast in the test and template images.
	Save the results in two vectors.
*/

void ObjectMatching::add_constrast(vector<Mat> tests, vector<Mat> templates, double alpha_test, double alpha_template)
{
	cout << "Adding contrast ..." << endl;

	int i;
	//Modify the test images
	for (i = 0; i < tests.size(); i++)
	{
		Mat img = tests[i].clone();

		int j;
		for (j = 0; j < tests[i].cols; j++)
		{
			int k;
			for (k = 0; k < tests[i].rows; k++)
			{
				img.at<Vec3b>(k, j)[0] = saturate_cast<uchar>(alpha_test * tests[i].at<Vec3b>(k, j)[0]);
				img.at<Vec3b>(k, j)[1] = saturate_cast<uchar>(alpha_test * tests[i].at<Vec3b>(k, j)[1]);
				img.at<Vec3b>(k, j)[2] = saturate_cast<uchar>(alpha_test * tests[i].at<Vec3b>(k, j)[2]);
			}
		}

		contrast_tests.push_back(img);
	}

	//Modify the template images
	for (i = 0; i < img_template.size(); i++)
	{
		Mat img = img_template[i].clone();

		int j;
		for (j = 0; j < img_template[i].cols; j++)
		{
			int k;
			for (k = 0; k < img_template[i].rows; k++)
			{
				img.at<Vec3b>(k, j)[0] = saturate_cast<uchar>(alpha_template * img_template[i].at<Vec3b>(k, j)[0]);
				img.at<Vec3b>(k, j)[1] = saturate_cast<uchar>(alpha_template * img_template[i].at<Vec3b>(k, j)[1]);
				img.at<Vec3b>(k, j)[2] = saturate_cast<uchar>(alpha_template * img_template[i].at<Vec3b>(k, j)[2]);
			}
		}
		contrast_templates.push_back(img);
	}
}

/**
	Function to apply the Bilater Filter to the test images, the parameters of the bilateral function are given in input.
	Save the result in a vector.
*/

void ObjectMatching::bilateral(vector<Mat> tests, int d, double sigma_color_test, double sigma_space_test)
{
	cout << "Applying Bilateral filter ... " << endl;

	int i;
	for (i = 0; i < tests.size(); i++)
	{
		Mat img = tests[i].clone();

		bilateralFilter(tests[i], img, d, sigma_color_test, sigma_space_test);

		filter_tests.push_back(img);
	}
}

/**
	Function to select the 10 best matches in the 60 found with the previous function.
	To select the best matches it's taken into account also the more frequency colors of the template selected and
	of the region in the test image where the template is matched.
	The function write the results in a file, associating each test image with the positions of the 10 best templates.
*/
void ObjectMatching::matching_selecting()
{
	cout << "Computing final selection of the best matches ..." << endl;
	String object_name;
	switch (object)
	{
	case 1:
		object_name = "can";
		break;
	case 2:
		object_name = "driller";
		break;
	case 3:
		object_name = "duck";
		break;
	}
	String file_name = object_name + "_result.txt";
	ofstream file(file_name);

	int i;
	for (i = 0; i < img_tests.size(); i++)
	{
		vector<int> vec_int;
		vector<double> best_values;
		vector<Point> vec_pnt;
		best_matches.push_back(vec_int);
		best_positions.push_back(vec_pnt);

		int j;
		//Compute the new score, based on the frequency colors, of the 60 best matches
		for (j = 0; j < best_sixty_templates[i].size(); j++)
		{
			vector<int> colors_template = frequently_color(img_template[best_sixty_templates[i][j]]);
			Mat submatrix = contrast_tests[i](Range(best_sixty_locations[i][j].y, 
											   best_sixty_locations[i][j].y+ img_template[best_sixty_templates[i][j]].rows), 
										 Range(best_sixty_locations[i][j].x, 
											   best_sixty_locations[i][j].x + img_template[best_sixty_templates[i][j]].cols)
										);
			Mat sub = submatrix.clone();
			int h;
			for (h = 0; h < img_template[best_sixty_templates[i][j]].cols; h++)
			{
				int k;
				for (k = 0; k < img_template[best_sixty_templates[i][j]].rows; k++)
				{
					if (img_masks[best_sixty_templates[i][j]].at<Vec3b>(k,h) == Vec3b(0, 0, 0))
						sub.at<Vec3b>(k,h) = Vec3b(0, 0, 0);
				}
			}
			
			vector<int> colors_position = frequently_color(sub);

			int score = 0;
			int k = 0;
			for (k = 0; k < colors_template.size(); k++)
			{
				score = score + 180 - abs(colors_template[k] - colors_position[k]);
			}
			best_sixty_values[i][j] = best_sixty_values[i][j] + score;
		}

		//Select the 10 best matches
		for (j = 0; j < best_sixty_templates[i].size(); j++)
		{
			if (j < 10)
			{
				best_matches[i].push_back(best_sixty_templates[i][j]);
				best_values.push_back(best_sixty_values[i][j]);
				best_positions[i].push_back(best_sixty_locations[i][j]);

				int k = j;
				while (k > 0 && best_values[k] > best_values[k - 1])
				{
					double temp1 = best_values[k - 1];
					best_values[k - 1] = best_values[k];
					best_values[k] = temp1;

					int temp2 = best_matches[i][k - 1];
					best_matches[i][k - 1] = best_matches[i][k];
					best_matches[i][k] = temp2;

					Point temp3 = best_positions[i][k - 1];
					best_positions[i][k - 1] = best_positions[i][k];
					best_positions[i][k] = temp3;

					k--;
				}
			}
			else if (best_sixty_values[i][j] > best_values[9])
			{
				best_values[9] = best_sixty_values[i][j];
				best_matches[i][9] = best_sixty_templates[i][j];
				best_positions[i][9] = best_sixty_locations[i][j];

				int k = 9;
				while (k > 0 && best_values[k] > best_values[k - 1])
				{
					double temp1 = best_values[k - 1];
					best_values[k - 1] = best_values[k];
					best_values[k] = temp1;

					int temp2 = best_matches[i][k - 1];
					best_matches[i][k - 1] = best_matches[i][k];
					best_matches[i][k] = temp2;

					Point temp3 = best_positions[i][k - 1];
					best_positions[i][k - 1] = best_positions[i][k];
					best_positions[i][k] = temp3;

					k--;
				}

			}
		}

		//Write the result in the output file
		file << "test" << i << ".jpg ";
		for (j = 0; j < best_matches[i].size(); j++)
			file << "model" << best_matches[i][j] << ".png " << best_positions[i][j].x <<" "<< best_positions[i][j].y << " ";

		file << endl;

	}
	file.close();

}

/**
	Function to show the position of the 10 best matches for each test image.
*/
void ObjectMatching::show_results()
{
	int i;
	for (i = 0; i < img_tests.size(); i++)
	{
		int height = img_tests[i].rows * 2;
		int width = img_tests[i].cols * img_tests.size() / 2;
		Mat results(height, width, img_tests[i].type());

		//Draw in a copy of the test image the template matched, in the correct position 
		int j;
		for (j = 0; j < best_matches.size(); j++)
		{
			Mat match = img_tests[i].clone();
			int h;
			for (h = 0; h < edges_templates[best_matches[i][j]].cols; h++)
			{
				int k;
				for (k = 0; k < edges_templates[best_matches[i][j]].rows; k++)
				{
					if (edges_templates[best_matches[i][j]].at<uchar>(k, h) > 240)
					{
						match.at<Vec3b>(k + best_positions[i][j].y, h + best_positions[i][j].x)[0] = 255;
						match.at<Vec3b>(k + best_positions[i][j].y, h + best_positions[i][j].x)[1] = 0;
						match.at<Vec3b>(k + best_positions[i][j].y, h + best_positions[i][j].x)[2] = 234;
					}
				}
			}

			Mat submatrix = results(Range( (j/5)*img_tests[i].rows, (j/5) * img_tests[i].rows + img_tests[i].rows), 
									Range(img_tests[i].cols * (j%5), img_tests[i].cols * (j%5) + img_tests[i].cols));
			match.copyTo(submatrix);
		}
		namedWindow("Final result", WINDOW_NORMAL | WINDOW_KEEPRATIO | WINDOW_GUI_EXPANDED);
		imshow("Final result", results);
		waitKey(0);
	}
	destroyAllWindows();
}


/**
	Function to computer the 2 more frequently color in the image given in input.
	Return, in a vector of int, the value of the H channel, in the HSV space, of the colors found.
*/

vector<int> frequently_color(Mat img)
{
	Mat hsv_img;
	cvtColor(img, hsv_img, COLOR_BGR2HSV);

	vector<int> values;
	int step = 10;
	int j;

	for (j = 0; j < 18; j++)
		values.push_back(0);

	for (j = 0; j < img.cols; j++)
	{
		int k;
		for (k = 0; k < img.rows; k++)
		{
			//Count the number of pixel for each colors (except the black)
			if (img.at<Vec3b>(k, j) != Vec3b(0, 0, 0))
			{
				//The H value are divided in range of 10 values each
				double index = floor((hsv_img.at<Vec3b>(k, j)[0] / (step * 1.0)));
				values[(int)index]++;
			}
		}
	}

	vector<int> colors;
	vector<int> count;
	int n_colors = 2;

	//Selecting the 2 more frequently colors
	for (j = 0; j < values.size(); j++)
	{
		if (j < n_colors)
		{
			colors.push_back(j * step);
			count.push_back(values[j]);

			int k = j;
			while (k > 0 && count[k] > count[k - 1])
			{
				int temp = count[k - 1];
				count[k - 1] = count[k];
				count[k] = temp;

				temp = colors[k - 1];
				colors[k - 1] = colors[k];
				colors[k] = temp;

				k--;
			}
		}
		else if (values[j] > count[n_colors - 1])
		{
			count[n_colors - 1] = values[j];
			colors[n_colors - 1] = j * step;

			int k = n_colors - 1;
			while (k > 0 && count[k] > count[k - 1])
			{
				int temp = count[k - 1];
				count[k - 1] = count[k];
				count[k] = temp;

				temp = colors[k - 1];
				colors[k - 1] = colors[k];
				colors[k] = temp;

				k--;
			}
		}
	}
	return colors;
}