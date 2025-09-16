package Recursion;

import java.util.Scanner;

class Demo8{
	static int convert(int n) {
		if(n == 0) {
			return 0;
		} else {
			return (n%2+10*convert(n/2));
		}
	}
}

public class BinaryToDecmal {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		int n = obj.nextInt();
		System.out.println(Demo8.convert(n));
		
	}
	
}
