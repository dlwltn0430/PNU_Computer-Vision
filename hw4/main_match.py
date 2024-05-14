import hw_utils as utils
import matplotlib.pyplot as plt

def main():
    # Test run matching with no ransac
    plt.figure(figsize=(20, 20))
    im = utils.Match('hw4/data/scene', 'hw4/data/basmati', ratio_thres=0.6)
    plt.title('Match')
    plt.imshow(im)

    # Test run matching with ransac //// part 1-2
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        'hw4/data/scene', 'hw4/data/book',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
    # plt.title('MatchRANSAC')
    plt.imshow(im)
    plt.axis('off')
    plt.savefig('1-1-1.png', bbox_inches='tight', pad_inches=0)

    # Test run matching with ransac //// part 1-2
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        'hw4/data/scene', 'hw4/data/box',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
    # plt.title('MatchRANSAC')
    plt.imshow(im)
    plt.axis('off')
    plt.savefig('1-1-2.png', bbox_inches='tight', pad_inches=0)

    # Test run matching with ransac //// part 1-2
    plt.figure(figsize=(20, 20))
    im = utils.MatchRANSAC(
        'hw4/data/library', 'hw4/data/library2',
        ratio_thres=0.6, orient_agreement=30, scale_agreement=0.5)
    # plt.title('MatchRANSAC')
    plt.imshow(im)
    plt.axis('off')
    plt.savefig('1-2.png', bbox_inches='tight', pad_inches=0)

if __name__ == '__main__':
    main()
