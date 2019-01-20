'''
Created on Oct 25, 2011
Modified on Oct 31, 2012
Modified on Mar 4, 2017 by Susan Rodger

@author: rcd, alexandrudutu, ola
'''
from PIL import Image
from PIL import ImageFilter
import math
# import cv2

# alternatively, try the line below
#import Image    # standard python image library


def apply_filter(image, zz, filter):
    '''
    Create and return a NEW image, based on a
    transform of each pixel in the given image using the given filter
    image is an open Image object
    filter is a function to apply to each pixel in image
    return new image, same size, to which filter has been applied to each pixel of image
    '''
    print(filter)
    if (zz == "b" or zz == "c"):
        z = int(input("By what percentage would you like to change it? (Please, refrain from using the percent sign!) "))/100
        pixels = [ filter(p, z) for p in image.getdata() ]
    # elif (zz == "e"):
    #     q = input("Please, give me the four coordinates, the area of which you wish to crop: ")
    #     pixels = cropp(image, q)
    else:
        pixels = [ filter(p) for p in image.getdata() ]
    nim = Image.new("RGB",image.size)
    nim.putdata(pixels)
    return nim

def open_image(filename):
    '''
    opens the given image and converts it to RGB format
    returns a default image if the given one cannot be opened
    filename is the name of a PNG, JPG, or GIF image file
    '''
    image = Image.open(filename)
    if image == None:
        print("Specified input file " + filename + " cannot be opened.")
        return Image.new("RGB", (400, 400))
    else:
        print(str(image.size) + " = " + str(len(image.getdata())) + " total pixels.")
        return image.convert("RGB")


'''
During this lab a pixel is a tuple of 3 values (R, G, B)
representing the red, green, and blue components of a color
that each range from [0-255] inclusive. 
The filter functions:
    - take one pixel as an argument,
    - modify the channels of the pixel and
    - return the transformed pixel
'''
def identity(pixel):
    '''
    returns a pixel that is unchanged from the original
    '''
    r,g,b = pixel
    return (r,g,b)
    

def invert(pixel):
    '''
    returns a pixel where every pixel channel is 255 minus its value
    '''
    r,g,b = pixel
    return (255-r, 255-g, 255-b)

def darken(pixel, z):
    """
    returns a pixel whose red, green, and blue values are all 90% of those given
    """
    # TODO: students fill this in
    # z = float(input("By what percentage would you like to brighten it by? (Please, refrain from using the percent sign!) "))/100
    r,g,b = pixel
    r = int(z*r)
    g = int(z*g)
    b = int(z*b)
    return (r, g, b)
 
def brighten(pixel, z):
    """
    returns a pixel whose red, green, and blue values are all 110% of those given
    but not over 255 (the maximum value).
    """
    # TODO: students fill this in
    # pass
    # r,g,b = pixel
    # return (int(1.1*r), int(1.1*g), int(1.1*b))
    # z = int(input("By what percentage would you like to brighten it by? (Please, refrain from using the percent sign!) "))/100
    r,g,b = pixel
    r = int(z*r)
    g = int(z*g)
    b = int(z*b)
    return (r, g, b)
    
def gray_scale(pixel):
    '''
    returns a pixel whose red, green, and blue values are all set to the same value: 
      the average of the original channels 
    '''
    # TODO: students fill this in
    r,g,b = pixel
    gr = int((r+g+b)/3)
    return (gr, gr, gr)

    
def posterize(pixel):
    """
    returns a pixel whose red, green, and blue values are all changed in
    the following way:
     - divide channel's range into 4 equal pieces (i.e., 0-63, 64-127, 128-191, 192-255)
     - set the channel's value to a fixed value within that range (i.e., 50, 100, 150, 200)
    """
    # TODO: students fill this in
    if (r <= 63 and g <=63 and b <= 63):
        return (50, 50, 50)
    elif (r <= 127 and g <=127 and b <= 127):
        return (100, 100, 100)
    elif (r <= 191 and g <=191 and b <= 191):
        return (150, 150, 150)
    else:
        return (200, 200, 200)

def solarize(pixel):
    """
    returns a pixel whose red, green, and blue values are all changed in
    the following way:
     - if the value is less than 128, set it to 255 - the original value.
     - if the value is 128 or greater, don't change it.
    """
    # TODO: students fill this in
    r,g,b = pixel
    if (r < 128):
        r = 255 - r
    if (g < 128):
        g = 255 - g
    if (b < 128):
        b = 255 - b
    return (r, g, b)

def denoise(pixel):
    '''
    take noise out of a pixel
    '''
    # TODO: students fill this in
    
    r,g,b = pixel
    return (r*10, 0, 0)

def denoise2(pixel):
    '''
    take noise out of a pixel
    '''
    # TODO: students fill this in
    r,g,b = pixel
    return (0, g*10, b*10)

def denoise3(pixel):
    '''
    take noise out of a pixel
    '''
    # TODO: Students fill this in
    r,g,b = pixel
    if (r == 255 and g == 0 and b == 0):
        return (0,0,0)
    elif (r == 255 and g==255 and b==255):
        return (0, 0, 0)
    else:
        return (r,g,b)



def cropp(pri, co, sl):
    img = Image.open(pri)
    cp = img.crop(co)
    cp.save(sl)
    cp.show()
    # image = "dukelogo.png"
    # cropp(image, (110, 120, 706, 1050), "cropped.png")


def load_and_go(fname,zz,filterfunc):
    image = open_image(fname)
    nimage = apply_filter(image,zz,filterfunc)
    # image.show()
    nimage.show()
    '''
    processedImage.jpg is the name of the file
    the image is saved in. The first time you do 
    this you may have to refresh to see it.
    '''
    nimage.save("processedImage.jpg")


if __name__ == "__main__":
    ''' Change the name of the file and the function
        to apply to the file in the line below
    '''
    def start():
        z = input("Insert your filename. Be sure to add the \".png\" or \".bmp\" at the end: ")
        image = open_image(z)
        image.show()
        print("Thank you! Your file should be shown now.")
        print("Now, what would you like to do?")
        zz = input(" Invert Colors (type a) \n Lighten (type b) \n Darken (type c) \n GrayScale (type d) \n Crop (type e) \n Blur (type f) \n Sharpen (type g) \n")
        if (zz == "a"):
            print("Please hold whilst we edit your file.")
            load_and_go(z, zz, invert)
            print("Like it?")
        elif (zz == "b"):
            print("Please hold whilst we edit your file.")
            load_and_go(z, zz, brighten)
            print("Like it?")
        elif (zz == "c"):
            print("Please hold whilst we edit your file.")
            load_and_go(z, zz, darken)
            print("Like it?")
        elif (zz == "d"):
            print("Please hold whilst we edit your file.")
            load_and_go(z, zz, gray_scale)
            print("Like it?")
        elif (zz == "e"):
            print("Please hold whilst we edit your file.")
            image = "dukelogo.png"
            i = 0
            ii = []
            while (i < 4):
                ii.append(float(input("Please insert a coordinate: ")))
                i+=1
            cropp(image, (ii[0], ii[1], ii[2], ii[3]), "cropped.png")
            print("Like it?")
        elif (zz == "f"):
            iiii = int(input("How big of a radius do you wish to have? "))
            # bl = z.filter(ImageFilter.BLUR)
            i = Image.open(z)
            ii = i.filter(ImageFilter.GaussianBlur(radius=iiii))
            ii.show()
            # i = Image.open(z)
            # ii = i.filter(ImageFilter.BLUR)
            # iii = i.filter(ImageFilter.MinFilter(3))
            # iiii = open_image(iii)

            # PIL.ImageFilter.GaussianBlue(radius=2)
            # img = cv2.imread(z)
            # blurImg = cv2.blur(img,(10,10))
            # cv2.imshow("blurred image", blurImg)
        elif (zz == "g"):
            iiii = int(input("How big of a radius do you wish to have? "))
            iiiii = int(input("How big of a percent do you wish to have? "))
            iiiiii = int(input("How big of a threshold do you wish to have? "))
            # bl = z.filter(ImageFilter.BLUR)
            i = Image.open(z)
            ii = i.filter(ImageFilter.UnsharpMask(radius=iiii, percent=iiiii, threshold=iiiiii))
            ii.show()
        elif (zz == "h"):
            img = Image.open(z)
            n = Image.new("RGB", (width, height), "white")
            ii = [(x-1, y-1), (x-1, y), (x-1, y+1), (x, y-1), (x, y+1), (x+1, y), (x+1, y+1)]
            iii = [-2*(r+g+b), -2*(r+g+b), 0, -2*(r+g+b), 2*(r+g+b), 0, 2*(r+g+b)]
            for x in range(1, width-1):
                for y in range(1, height-1):
                    ix = 0
                    iy = 0
                    while (i < len(ii)):
                        ip = img.getpixel(ii[i])
                        r = ip[0]
                        g = ip[1]
                        b = ip[2]
                        # intense = r+g+b
                        ix-= iii[i]
                        iy -= iii[i]
                        i+=1
            img.show()
        else:
            print("Invalid")
            
        # input_file = "dukelogo.png"
        # # load_and_go(input_file, invert)
        # # load_and_go(input_file, darken)
        # # # load_and_go(input_file, brighten)
        # # load_and_go(input_file, gray_scale)
        # # load_and_go(input_file, solarize)
        # # load_and_go(input_file, posterize)
        # input_file = "noise.png"
        # load_and_go(input_file, denoise)
        # input_file = "copper-puzzle.png"
        # load_and_go(input_file, denoise2)
        # input_file = "clue2.bmp"
        # load_and_go(input_file, denoise3)
        # input_file = "clue.bmp"
        # load_and_go(input_file, denoise3)
        again = input("Would you like to go again? ").lower().strip()
        if (again == "yes"):
            start()
    start()