These files are used to create dev and test sets. You don't need to use them since the sets are already provided to you. Use them only if you want to create new dev and test files to play more with code development. 

To create new dev and test files, first download the 20news data set here: http://qwone.com/~jason/20Newsgroups/

Then run ./convert2utf8.sh 20news-18828
(Note: I used iconv for conversion on Ubuntu. For Mac, there may be a different program which does the conversion.)

Finally run python extract_dev_and_test_set.py 20news-18828
This creates a new folder called extractedFiles that contains the dev and test set as well as dev_keys.txt and test_keys.txt. These files map each file in the dev/test set to its true group. The files in the dev/test set have been removed from the 20newsgroup-18828 folder which can now be used as training data.
