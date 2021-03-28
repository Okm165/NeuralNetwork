from matplotlib import pyplot as plt

width, height = 28, 28
counter = 0
row = 0
imageMatrix = [[0 for x in range(width)] for y in range(height)]
plt.figure(1)
file = open("8_3.pgm", "rb")
for k in range(height):
    for j in range(width):
        imageMatrix[k][j] = (255-ord(file.read(1)))/255
plt.subplot(331)
plt.title("original")
plt.imshow(imageMatrix)
filterArray = []

# filter = [[1,1,1,1,1],[0,0,0,0,0],[0,0,0,0,0],[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1]]
# filterArray.append(filter)
filter = [[-1,-1,-1,-1,-1],[-1,-1,-1,-1,-1],[0,0,0,0,0],[0,0,0,0,0],[1,1,1,1,1]]
filterArray.append(filter)
filter = [[1,0,0,-1,-1],[1,0,0,-1,-1],[1,0,0,-1,-1],[1,0,0,-1,-1],[1,0,0,-1,-1]]
filterArray.append(filter)
filter = [[-1,-1,0,0,1],[-1,-1,0,0,1],[-1,-1,0,0,1],[-1,-1,0,0,1],[-1,-1,0,0,1]]
filterArray.append(filter)
filter = [[-1,0,1,0,-1],[0,1,0,1,0],[-1,0,-1,0,-1],[0,1,0,1,0],[-1,0,1,0,-1]]
filterArray.append(filter)

imageFilteredMatrix = [[0 for x in range(width-len(filter)+1)] for y in range(height-len(filter[0])+1)]
for z in range(len(filterArray)):
    for k in range(len(imageFilteredMatrix)):
        for j in range(len(imageFilteredMatrix[0])):
            sum = 0
            for lokalK in range(len(filterArray[z])):
                for lokalJ in range(len(filterArray[z][0])):
                    sum += imageMatrix[k+lokalK][j+lokalJ] * filterArray[z][lokalK][lokalJ]
            imageFilteredMatrix[k][j] = max(0, sum)
    
    plt.subplot(332+z)
    plt.title("filter: " + str(z))
    plt.imshow(imageFilteredMatrix)

    step = 6
    imageMaxPooledMatrix = [[0 for x in range((width-len(filter)+1)//step)] for y in range((height-len(filter[0])+1)//step)]
    for k in range(0, len(imageFilteredMatrix), step):
        for j in range(0, len(imageFilteredMatrix[0]), step):
            values = []
            for lokalK in range(step-1):
                for lokalJ in range(step-1):
                    values.append(imageFilteredMatrix[k+lokalK][j+lokalJ])
            imageMaxPooledMatrix[k//step][j//step] = max(values)
    plt.subplot(332+z+4)
    plt.title("maxpooling: ")
    plt.imshow(imageMaxPooledMatrix)
plt.subplots_adjust(left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.3)



plt.show()