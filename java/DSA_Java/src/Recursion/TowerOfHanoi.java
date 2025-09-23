package Recursion;

import java.util.Scanner;

class Demo21{
	static void towerOfHanoi(int n, String src, String helper, String dest) {
		if(n==1) {
			System.out.println("Move The Disk "+n+"  from "+src+"to "+dest);
			return;
		}
		towerOfHanoi(n-1,src,dest,helper);
		System.out.println("Move The Disk "+n+"from "+src+"to "+dest);
		towerOfHanoi(n-1,helper,src,dest);
	}
}

public class TowerOfHanoi {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter number of disks:");
		
		int n = obj.nextInt();
		
		Demo21.towerOfHanoi(n, "s", "h", "d");
		
	}
	
}
