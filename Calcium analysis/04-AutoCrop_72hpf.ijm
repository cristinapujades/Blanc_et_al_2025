
#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "File suffix", value = ".tif") suffix

processFolder(input);
function processFolder(input) {
	list = getFileList(input);
	list = Array.sort(list);
	for (i = 0; i < list.length; i++) {
		if(File.isDirectory(input + File.separator + list[i]))
			processFolder(input + File.separator + list[i]);
		if(endsWith(list[i], suffix))
			processFile(input, output, list[i]);
	}
}

function processFile(input, output, file) {
	    open(input + File.separator + file);
  		selectWindow(File.name);
  		    //Apply metadata
    Stack.setXUnit("micron");
    run("Properties...", "channels=1 slices=192 frames=1 pixel_width=0.5681825 pixel_height=0.5681825 voxel_depth=1.0404596");
  		makeRectangle(64, 0, 689, 634);
		run("Crop");
		saveAs("Tiff", output + File.separator + File.nameWithoutExtension + ".tif");
		close();  // Close the image after saving
		run("Close All");  // Close all remaining open images
		call("java.lang.System.gc");
} 
	