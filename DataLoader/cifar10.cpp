#include "cifar10.h"

#include <sys/types.h>
#include <sys/stat.h>

using namespace std;
using namespace torch;


CIFAR10::CIFAR10(const std::string& root, bool train)
{
	const int ImageRows = 32;
	const int ImageCols = 32;
	const int ImageChannels = 3;
	if (train) {
		// Expected files to open
		vector<string> idx = {"1","2","3","4","5"};
		vector<string> train_file;
		for (auto i : idx) {
			train_file.push_back("/cifar-10-batches-bin/data_batch_"+i+".bin");
		}

		// Check if root path is valid
		bool files_downloaded = true;
		bool error = false;
		struct stat info;
		if( stat( root.c_str(), &info ) != 0 ) { 
			printf( "cannot access %s\n", root.c_str() );
			error = true;
		}
		else if( !( info.st_mode & S_IFDIR) ) {  // S_ISDIR() doesn't exist on my windows 
			printf( "%s is no directory\n", root.c_str() );
			error = true;
    	}
    	if (!error) {

    		// Check for the existence of the binary files
    		if( stat( (root+"/cifar-10-batches-bin").c_str(), &info ) != 0 ) { 
				files_downloaded = false;
			} else {
				for (auto file : train_file) {
					if( stat( (root+file).c_str(), &info ) != 0 ) { 
						files_downloaded = false;
					}
				}
			}
			// Download files if not found
			if (!files_downloaded) {
				cout << "CIFAR10 files not found" << endl;
				if( stat( (root+"/cifar-10-binary.tar.gz").c_str(), &info ) != 0 ) { 
					cout << "Downloading CIFAR10 files" << endl;
					// Wraps path with "" toa avoid errors
					system( ("wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz -P "+string("\"")+root+string("\"")).c_str() );
				}
				cout << "Extracting CIFAR10 files" << endl;
				//printf((string("tar xf \"")+root+string("/cifar-10-binary.tar.gz\"")).c_str());
				//cout << "Extracting" << endl;
				//system( ("cd \""+root+"\"").c_str() );
				string tar_command =  "tar xf "+string("\"")+root+string("/cifar-10-binary.tar.gz\" -C "+string("\"")+root+string("\""));
				system(tar_command.c_str() );
			
			} else {
				cout << "CIFAR10 files already downloaded" << endl;
			}

			for (auto file : train_file) {
				const auto path = root + file;
				std::ifstream bin_file(path, std::ios::binary);
				TORCH_CHECK(bin_file, "Error opening images file at ", path);

				bool reading = true;
				do {
					auto label = torch::empty(1, torch::kByte);
					auto image = torch::empty({ ImageChannels, ImageRows, ImageCols}, torch::kByte);
					reading = (bin_file.read(reinterpret_cast<char*>(label.data_ptr()), 1) &&
						bin_file.read(reinterpret_cast<char*>(image.data_ptr()), image.numel()) );
					image = image.to(torch::kFloat32).div_(255);
					labels.push_back(label);
					images.push_back(image);
				} while (reading);
			}
			cout << labels.size() << "   "  << images.size() << endl;
		}
	} else {
		const string test_file = "/test_batch.bin";
	}
	
}