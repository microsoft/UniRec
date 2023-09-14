
import argparse
import os

r''' 
    Process data file: the last item is test set, the second last item is valid set, the rest are train set.
'''
def run(args):
    os.makedirs(args.outdir, exist_ok=True)
    train_file = os.path.join(args.outdir, 'train.txt')
    valid_file = os.path.join(args.outdir, 'valid.txt')
    test_file = os.path.join(args.outdir, 'test.txt')
    user_history_file = os.path.join(args.outdir, 'user_history.txt')
    
    wt_train = open(train_file, 'w')
    wt_valid = open(valid_file, 'w')
    wt_test =  open(test_file, 'w')
    wt_user_history = open(user_history_file, 'w')
    lengths = []
    with open(args.infile, 'r') as rd:
        for line in rd.readlines():
            words = line.strip().split(' ')
            userid, items = words[0], words[1:]
            item_set = set()
            dedup_items = []
            for item in items:
                if item not in item_set:
                    item_set.add(item)
                    dedup_items.append(item)
            items = dedup_items
            lengths.append(len(items))
            wt_train.write(userid + ' ' + ','.join(items[:-2]) + '\n')
            wt_valid.write(userid + ' ' + items[-2] + '\n')
            wt_test.write(userid + ' ' + items[-1] + '\n')
            wt_user_history.write(userid + ' ' + ','.join(items) + '\n')
            
    wt_train.close()
    wt_valid.close()
    wt_test.close()
    wt_user_history.close()
    print(f'max item length: {max(lengths)}')
    print(f'min item length: {min(lengths)}')

def main():
    arguments = argparse.ArgumentParser()
    arguments.add_argument('--infile', type=str)
    arguments.add_argument('--outdir', type=str)
    
    args = arguments.parse_args()
    
    run(args)

if __name__ == '__main__':
    main()