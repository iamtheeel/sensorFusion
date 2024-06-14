/*
  This sketch reads a raw Stream of RGB565 pixels
  from the Serial port and displays the frame on
  the window.

  Use with the Examples -> CameraCaptureRawBytes Arduino sketch.

  This example code is in the public domain.
*/

import processing.serial.*;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;

Serial myPort;

// must match resolution used in the sketch
final int cameraWidth = 96;
final int cameraHeight = 96;
final int cameraBytesPerPixel = 2; //RGB565 is 2bytes/pixle
final int bytesPerFrame = cameraWidth * cameraHeight * cameraBytesPerPixel;

PImage myImage;
byte[] frameBuffer = new byte[bytesPerFrame];

void setup()
{
  size(94, 94);
  //myPort = new Serial(this, "/dev/cu.usbserial-11310", 921600);     // Mac
  myPort = new Serial(this, "/dev/ttyUSB1", 1000000);     // Lunux

  // wait for full frame of bytes
  myPort.buffer(bytesPerFrame);  

  myImage = createImage(cameraWidth, cameraHeight, RGB);
}

void draw()
{
  image(myImage, 0, 0);
}

void serialEvent(Serial myPort) {
  // read the saw bytes in
  //println("Got Data");
  int input = myPort.read();
  if(input == 'A')
  {
    println("Got Data start");
    input = myPort.read();
    if(input == '3')
    {
    input = myPort.read();
    if(input == '3')
    {
    println("Start img stream");
    myPort.readBytes(frameBuffer);
  
    // access raw bytes via byte buffer
    ByteBuffer bb = ByteBuffer.wrap(frameBuffer);
    //bb.order(ByteOrder.BIG_ENDIAN);
    bb.order(ByteOrder.LITTLE_ENDIAN);

  
    int i = 0;
  
    while (bb.hasRemaining()) {
      // read 16-bit pixel
      short p = bb.getShort();
  
      // convert RGB565 to RGB 24-bit
      // (R:HB7-HB3, G:HB2-LB5, B: LB4-LB0)
      int r = ((p >> 11) & 0x1f) << 3;
      int g = ((p >> 5) & 0x3f) << 2;
      int b = ((p >> 0) & 0x1f) << 3;
  
      // set pixel color
      myImage .pixels[i++] = color(r, g, b);
    }
   myImage .updatePixels();
  } // A
  } // 3
  } // 3

}
