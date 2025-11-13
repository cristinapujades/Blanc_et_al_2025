// this macro allows to make the average between 5 samples for two channels UserAveraged (here after threshold)
// the input is named UserAveragedC(channel number)E(sample number).tif
// it ouputs stacks named SampleAveragedC(channel number).tif

#@ File (label = "Input directory", style = "directory") input
#@ File (label = "Output directory", style = "directory") output
#@ String (label = "samples", value = "10") samples

for (i = 1; i < samples+1; i++) {
	open(input + File.separator + "C2E" + i + ".tif");
	setThreshold(1, 65535);
	run("Convert to Mask", "background=Dark black");
	open(input + File.separator + "C3E" + i + ".tif");
	setThreshold(1, 65535);
	run("Convert to Mask", "background=Dark black");
	}
run("Merge Channels...", "c1=" + "C3E1.tif c2=" + "C3E2.tif c3=" + "C3E3.tif c4=" + "C3E4.tif c5=" + "C3E5.tif create c6=" + "C3E6.tif c7=" + "C3E7.tif c8=" + "C3E8.tif c9=" + "C3E9.tif c10=" + "C3E10.tif create");
run("Re-order Hyperstack ...", "channels=[Frames (t)] slices=[Channels (c)] frames=[Slices (z)]");
run("Z Project...", "projection=[Average Intensity] all");
run("Re-order Hyperstack ...", "channels=[Channels (c)] slices=[Frames (t)] frames=[Slices (z)]");
saveAs("Tiff", output + File.separator + "SampleAveragedC3.tif");
run("Merge Channels...", "c1=" + "C2E1.tif c2=" + "C2E2.tif c3=" + "C2E3.tif c4=" + "C2E4.tif c5=" + "C2E5.tif create c6=" + "C2E6.tif c7=" + "C2E7.tif c8=" + "C2E8.tif c9=" + "C2E9.tif c10=" + "C2E10.tif create");
run("Re-order Hyperstack ...", "channels=[Frames (t)] slices=[Channels (c)] frames=[Slices (z)]");
run("Z Project...", "projection=[Average Intensity] all");
run("Re-order Hyperstack ...", "channels=[Channels (c)] slices=[Frames (t)] frames=[Slices (z)]");
saveAs("Tiff", output + File.separator + "SampleAveragedC2.tif");

close("*");
