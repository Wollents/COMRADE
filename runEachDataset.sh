echo -e "\033[31mrunning the Cora\033[0m"
python main.py --dataset cora --gama 0.6 --beta 0.9
echo -e "\033[31mrunning the Citeseer\033[0m"
python main.py --dataset citeseer --gama 0.6 --beta 0.8
echo -e "\033[31mrunning the Pubmed\033[0m"
python main.py --dataset pubmed --gama 0.6 --beta 0.8
echo -e "\033[31mrunning the Flickr\033[0m"
python main.py --dataset Flickr --gama 0.4 --beta 0.1
echo -e "\033[31mrunning the DBLP\033[0m"
python main.py --dataset dblp --gama 0.6 --beta 0.6

