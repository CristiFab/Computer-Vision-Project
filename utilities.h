#include <opencv2/core/utils/filesystem.hpp>
#include <fstream>

/**
	Implementation of my class to manage the pre-processing of the image,
	the computation of the template matching and the refinement of
	the results.
*/

class ObjectMatching {

public:

	ObjectMatching(std::vector< cv::Mat > test_images, std::vector< cv::Mat > masks, std::vector< cv::Mat > templates, int obj );
	void load_information(cv::String path);
	void bilateral(std::vector<cv::Mat> tests, int d, double sigma_color_test, double sigma_space_test);
	void add_constrast(std::vector<cv::Mat> tests, std::vector<cv::Mat> templates, double alpha_test, double alpha_template);
	void edge_detection(std::vector<cv::Mat> tests, std::vector<cv::Mat> templates, int test_threshold1, int test_threshold2, int template_threshold1, int template_threshold2);
	void template_matching();
	void matching_selecting();
	void show_results();

public:
	std::vector<cv::Mat> img_tests, img_masks, img_template,
		edges_tests, edges_templates,
		contrast_tests, contrast_templates,
		filter_tests; 
	int object, size;
	std::vector<std::vector<cv::Point>> best_sixty_locations, best_positions;
	std::vector<std::vector<int>> best_sixty_templates, best_matches;
	std::vector<std::vector<double>> best_sixty_values;
	
};

/**
	Function to compute the most frequent colors in an image
*/
std::vector<int> frequently_color(cv::Mat img);