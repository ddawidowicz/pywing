from load_data import load_data

def main():
    data1()
    data2()


def data1():
    infile = 'data/data1_exam_scores.csv'
    mat = False
    csv = True
    tab = False

    data = load_data(infile, mat=mat, csv=csv, tab=tab)
    x = data[:,[0,1]]
    y = data[:, [2]]
    pos = np.where(y==1)[0] #get the indices of the positive class labels
    neg = np.where(y==0)[0] #get the indices of the negative class labels

    plt.plot(x[pos,0],x[pos,1],'bs',linewidth=2.0,label='Admitted')
    plt.plot(x[neg,0],x[neg,1],'ro',linewidth=2.0,label='Declined')
    plt.axis([25,110, 25, 110])
    plt.xlabel('Exam 1')
    plt.ylabel('Exam 2')
    plt.title('Data Set 1 - Exam Scores')
    plt.legend(numpoints=1)
    plt.show()

def data2():
    infile = 'data/data2_microchip.csv'
    mat = False
    csv = True
    tab = False

    data = load_data(infile, mat=mat, csv=csv, tab=tab)
    x = data[:,[0,1]]
    y = data[:, [2]]
    pos = np.where(y==1)[0] #get the indices of the positive class labels
    neg = np.where(y==0)[0] #get the indices of the negative class labels

    plt.plot(x[pos,0],x[pos,1],'bs',linewidth=2.0,label='Passed')
    plt.plot(x[neg,0],x[neg,1],'ro',linewidth=2.0,label='Failed')
    plt.axis([-1, 1.5, -0.8, 1.2])
    plt.xlabel('Microchip 1 Test')
    plt.ylabel('Microchip 2 Test')
    plt.title('Data Set 2 - Microchip Testing')
    plt.legend(numpoints=1)
    plt.show()

main()
