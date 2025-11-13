requires("1.33s"); 
dir = getDirectory("Input Directory ");
output = getDirectory("Directory for output");
setBatchMode(true);
count = 0;
countFiles(dir);
n = 0;
processFiles(dir);
print(count+" files processed");
print("Saved to: " + output);
   
function countFiles(dir) {
    list = getFileList(dir);
    for (i=0; i<list.length; i++) {
        if (endsWith(list[i], "/"))
            countFiles(""+dir+list[i]);
        else if (endsWith(list[i], "CHN00.filtered_100.tif"))
            count++;
    }
}

function processFiles(dir) {
    list = getFileList(dir);
    for (i=0; i<list.length; i++) {
        if (endsWith(list[i], "/"))
            processFiles(""+dir+list[i]);
        else {
            path = dir+list[i];
            if (endsWith(path, "CHN00.filtered_100.tif")) {
                showProgress(n++, count);
                processFile(path);
                }
            }
        }

}


function processFile(path) {
    print("Processing: " + path);
    if (File.exists(path)) {
        open(path);
        selectWindow(File.getName(path));
        run("Remove Outliers...", "radius=1 threshold=50 which=Bright stack");
        run("Properties...", "channels=1 slices=92 frames=1 unit=micron pixel_width=0.40625 pixel_height=0.40625 voxel_depth=2.2");
        run("Scale...", "x=0.7154391414730302 y=0.7154391414730302 z=2.114450191050186 width=869 height=643 depth=195 interpolation=Bicubic process create");
        run("Flip Horizontally", "stack");
        makeRectangle(70, 0, 786, 643);
        run("Crop");
        saveAs("Tiff", output + File.separator + File.getName(path) + ".tif");
        close("*");
        run("Collect Garbage");
	    call("java.lang.System.gc");
        print(File.getName(path) + " - done");
    }
}