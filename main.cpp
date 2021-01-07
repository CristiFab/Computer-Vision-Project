#include<opencv2/opencv.hpp>
#include<iostream>
#include "utilities.h"

using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	while (true)
	{
		cout << "******************************************************************" << endl;
		cout << "Choose the object and digit the corresponding number or 0 to exit: " << endl;
		cout << "1) can" << endl << "2) driller" << endl << "3) duck" << endl;
		cout << "******************************************************************" << endl;

		int obj = 0;
		cin >> obj;
		String obj_name;
		int template_thr1 = 0;
		int template_thr2 = 0;
		int test_thr1 = 0;
		int test_thr2 = 0;
		double alpha_test = 0.0;
		double alpha_template = 0.0;
		double sigma_space_test = 0.0;
		double sigma_color_test = 0.0;
		int d = 0;

		switch (obj)
		{
		case 0:
			return 0;
		case 1:
			obj_name = "can";
			test_thr1 = test_thr2 = 150;
			template_thr1 = template_thr2 = 80;
			alpha_test = 0.5;
			alpha_template = 1;
			sigma_space_test = 100;
			sigma_color_test = 100;
			d = 5;
			break;
		case 2:
			obj_name = "driller";
			test_thr1 = 120;
			test_thr2 = 100;
			template_thr1 = 100;
			template_thr2 = 60;
			alpha_test = 2.5;
			alpha_template = 1;
			sigma_space_test = 100;
			sigma_color_test = 100;
			d = 9;
			break;
		case 3:
			obj_name = "duck";
			test_thr1 = 63;
			test_thr2 = 25;
			template_thr1 = 30;
			template_thr2 = 25;
			alpha_test = 1;
			alpha_template = 4;
			sigma_space_test = 70;
			sigma_color_test = 70;
			d = 3;
			break;
		default:
			cout << "Not valid number" << endl;
			return 0;
		}

		vector<Mat> test_images, masks, templates;

		ObjectMatching object(test_images, masks, templates, obj);
		object.load_information("data/" + obj_name);
		object.bilateral(object.img_tests, d, sigma_color_test, sigma_space_test);
		object.add_constrast(object.filter_tests, object.img_template, alpha_test, alpha_template);
		object.edge_detection(object.contrast_tests, object.contrast_templates, test_thr1, test_thr2, template_thr1, template_thr2);
		object.template_matching();
		object.matching_selecting();
		//object.show_results();
		cout << "Match done" << endl << endl;
	}

	return 0;
}
