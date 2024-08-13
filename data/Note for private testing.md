To run the prediction:
1. Set the private test dataset path in code/main.py as 
	data_dir = ".../.../private_test_images_2024.npy"
2. In code/main.py, add
	from DataLoader import load_testing_images
3. In code/DataLoader.py, complete the function load_testing_images. The output of the function, x_test, should have shape (2000, 3072).
4. Run the following commands (with GPU)
	cd ./code
	python main.py --mode predict 
5. Store your prediction results as an array into an .npy file named “predictions.npy”. For each image, store a vector of the probabilities for the 10 classes instead of the predicted class. The shape of the saved array should be [2000, 10].