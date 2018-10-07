#include "clus_str.h"
#include "arraylist.h"

ClusterBank::ClusterBank(const rMatrix3d &cell,
              const Array<rVector3d> &atom_pos,
              int ntuple,
              const SpaceGroup &_equivalent_by_symmetry):
  cluster_list(), current_cluster(),
  current_index(0), previous_length(0.), multiplet(cell,atom_pos,ntuple),
  equivalent_by_symmetry(_equivalent_by_symmetry) {
    if (ntuple>1) {
      //printf("BREAKING POINT N5\n");
      //cout <<"cluster_list size (1): " <<cluster_list.getSize() << endl;
      cluster_list << new Cluster(multiplet);
      //cout <<"cluster_list size (1): " <<cluster_list.getSize() << endl;
      //printf("BREAKING POINT N6\n");
    }
    else {
       for (int at=0; at<atom_pos.get_size(); at++) {
         Cluster point(1);
         point(0)=atom_pos(at);
         add_unique(&cluster_list,point,equivalent_by_symmetry);
       }
    }
    current_cluster.init(cluster_list);
    //cout << " current_cluster.init breaking point " <<endl;
}

ClusterBank::ClusterBank(const ClusterBank &clusterbank,int ntuple):
  cluster_list(), current_cluster(),
  current_index(0),
  previous_length(0.),
  //printf("BREAKING POINT N7\n");
  multiplet(clusterbank.multiplet,ntuple),
  //printf("BREAKING POINT N8\n");
  equivalent_by_symmetry(clusterbank.equivalent_by_symmetry) {
    //cout <<"cluster_list size (2): " <<cluster_list.getSize() << endl;
    cluster_list << new Cluster(multiplet);
    //cout <<"cluster_list size (2): " <<cluster_list.getSize() << endl;
    current_cluster.init(cluster_list);
}

void ClusterBank::reset(void) {
  current_index=0;
  current_cluster.init(cluster_list);
}

void ClusterBank::operator++(int) {
  //cout << "BREAKING POINT N1" << endl;
  previous_length=get_length_quick(*current_cluster);
  //cout << "BREAKING POINT N2" << endl;
  LinkedListIterator<Cluster> save=current_cluster;
  //cout << "BREAKING POINT N3" << endl;
  current_cluster++;
  //cout << "BREAKING POINT N4" << endl;
  current_index++;
  if (!current_cluster && save->get_size()>1) {
    do {
       //cout << "NEW BREAKING POINT 0" << endl;
       //cout << cluster_list.getSize() <<endl;
       //cout << multiplet.cell << endl;
       //cout <<" Stucked here???"<<endl; 
       multiplet++;
       //ZSF debugging
       //if {multiplet(i)}
       for (int i=0; i< save->get_size(); i++) {
         //cout <<"multiplet: " << multiplet(i) <<endl;
       }
       //cout << "NEW BREAKING POINT 1" << endl; 
       } while (!add_unique(&cluster_list,Cluster(multiplet),equivalent_by_symmetry));
    //cout << "--------------------------"<< endl;
    //cout << "This cluster is unique!!!" <<endl;
    current_cluster=save;
    current_cluster++;
  }
}

MultiClusterBank::MultiClusterBank(const Structure &_lat,
				   int ntuple,
				   const SpaceGroup &_equivalent_by_symmetry):
  lat(_lat), cluster_bank(_lat.cell,_lat.atom_pos,ntuple,_equivalent_by_symmetry),
    cluster_list(), current_cluster(),
    current_index(0), previous_length(0.), 
    equivalent_by_symmetry(_equivalent_by_symmetry) {
  if (ntuple>1) {
    //cout << " make_new_clusters breaking point 0 " <<endl;
    make_new_clusters();
    //cout << " make_new_clusters breaking point 1 " <<endl;
  }
  else {
    //cout << "1-Point cluster generating:" << endl;
    //cout << "atom_pos.size: " << lat.atom_pos.get_size() <<endl;
    for (int at=0; at<lat.atom_pos.get_size(); at++) {
      //cout <<"point.site_type: "<<lat.atom_type(at)-2 <<endl;
      MultiCluster point(1);
      point.clus(0)=lat.atom_pos(at);
      //cout <<"point.clus: "<<lat.atom_pos(at) <<endl;
      point.site_type(0)=(lat.atom_type(at)-2);
      for (int t=0; t<lat.atom_type(at)-1; t++) {
	point.func(0)=t;
	add_unique(&cluster_list,point,equivalent_by_symmetry);
      }
    }
  //cout << "1-Point cluster size: " << cluster_list.getSize() <<endl;
  }
  current_cluster.init(cluster_list);
  //cout << " current_cluster.init breaking point 1" <<endl;
}

MultiClusterBank::MultiClusterBank(const MultiClusterBank &bank, int ntuple):
  lat(bank.lat), cluster_bank(bank.lat.cell,bank.lat.atom_pos,ntuple,bank.equivalent_by_symmetry),
    cluster_list(), current_cluster(),
    current_index(0), previous_length(0.), 
    equivalent_by_symmetry(bank.equivalent_by_symmetry) {
  make_new_clusters();
  current_cluster.init(cluster_list);
  //cout << " current_cluster.init breaking point 2" <<endl;
}

void MultiClusterBank::reset(void) {
  current_index=0;
  current_cluster.init(cluster_list);
}

void MultiClusterBank::make_new_clusters(void) {
  MultiCluster new_clus;
  //cout << "make_new_clusters breaking point 0" << endl;
  new_clus.clus=cluster_bank;
  //cout << "make_new_clusters breaking point 1" << endl;
  new_clus.site_type.resize(new_clus.clus.get_size());
  new_clus.func.resize(new_clus.clus.get_size());
  Array<int> minconfig(new_clus.clus.get_size());
  zero_array(&minconfig);
  rMatrix3d inv_cell=!(lat.cell);
  for (int at=0; at<new_clus.clus.get_size(); at++) {
    new_clus.site_type(at)=lat.atom_type(which_atom(lat.atom_pos,new_clus.clus(at),inv_cell))-2;
  }
  LinkedList<MultiCluster> new_clus_list;
  MultiDimIterator<Array<int> > config(minconfig,new_clus.site_type);
  for (; config; config++) {
    new_clus.func=config;
    add_unique(&new_clus_list,new_clus,equivalent_by_symmetry);
  }
  transfer_list(&cluster_list,&new_clus_list);
}

void MultiClusterBank::operator++(int) {
  //cout << "MultiClusterBank::operator++ breaking point 0" <<endl;
  previous_length=get_length_quick(current_cluster->clus);
  //cout <<"previous_length: " << previous_length <<endl;
  for (int i=0; i<current_cluster->clus.get_size(); i++) {
    //cout <<"current_cluster: " << current_cluster->clus(i) <<endl;
  }
  int temp_size = current_cluster->clus.get_size();
  //cout << "MultiClusterBank::operator++ breaking point 1" <<endl;
  //cout << cluster_list.getSize() << endl;
  current_cluster++;
  //cout << "MultiClusterBank::operator++ breaking point 2" <<endl;
  current_index++;
  /*for (int i=0; i<current_cluster->clus.get_size(); i++) {
    cout <<"After_++_cluster: " << current_cluster->clus(i) <<endl;
  }*/
  //cout << "MultiClusterBank::operator++ breaking point 3" <<endl;
  if (!current_cluster && temp_size != 1 ) {//After modify find_unitcell, here '&& previous_length!=0' may also need to be removed. And add new describenation!!!
    //cout << "cluster_bank++ breaking point 0" <<endl;
    cluster_bank++; // so the cluster_bank here shall not contain 1-point cluster? Or maybe just leaveout those empty clusters
    //cout << "cluster_bank++ breaking point 1" <<endl;
    make_new_clusters();
    //cout << "cluster_bank++ breaking point 2" <<endl;
  }
}


