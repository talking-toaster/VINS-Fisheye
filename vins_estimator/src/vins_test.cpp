#include <ros/ros.h>
#include <memory>
#include <opencv2/opencv.hpp>
#include "utility/queue_wrapper.hpp"

class MyResource {


  public:
	int id;
	// 构造函数
	MyResource(int _id) : id(_id) {
		std::cout << "Constructor called for MyResource with ID: " << id << std::endl;
	}

	// 移动构造函数
	MyResource(MyResource &&other) noexcept : id(other.id) {
		other.id = -1; // 禁用源对象的ID
		std::cout << "Move Constructor called for MyResource with ID: " << id << std::endl;
	}

	// 拷贝构造函数
	MyResource(const MyResource &other) : id(other.id) {
		std::cout << "Copy Constructor called for MyResource with ID: " << id << std::endl;
	}

	// 移动赋值操作符
	MyResource &operator=(MyResource &&other) noexcept {
		if (this != &other) {
			id		 = other.id;
			other.id = -1; // 禁用源对象的ID
			std::cout << "Move Assignment Operator called for MyResource with ID: " << id << std::endl;
		}
		return *this;
	}

	// 拷贝赋值操作符
	MyResource &operator=(const MyResource &other) {
		if (this != &other) {
			id = other.id;
			std::cout << "Copy Assignment Operator called for MyResource with ID: " << id << std::endl;
		}
		return *this;
	}

	// 析构函数
	~MyResource() {
		std::cout << "Destructor called for MyResource with ID: " << id << std::endl;
	}
};
std::shared_ptr<cv::Mat> getImg() {
	cv::Mat					 img(640, 480, CV_8UC1);
	std::shared_ptr<cv::Mat> img_ptr = std::make_shared<cv::Mat>(img);
	return img_ptr;
}

int main(int argc, char **argv) {
	ros::init(argc, argv, "vin_test");
	ros::NodeHandle n("~");

	RW_Queue<std::shared_ptr<MyResource>> queue(10);
	{
		MyResource					resource(1);
		std::shared_ptr<MyResource> ptr = std::make_shared<MyResource>(resource);
		queue.push(ptr);
		ptr = std::make_shared<MyResource>(MyResource(2));
	}
	std::shared_ptr<MyResource> ptr;
	queue.try_pop(ptr);
	std::cout << "ptr id: " << ptr->id << std::endl;
	std::cout << "use count: " << ptr.use_count() << std::endl;
}