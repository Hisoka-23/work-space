package Recursion;

import java.util.Scanner;

//01. Implement a program to print natural numbers from 1 to n
class Demo1{
	static void print(int n) {
		if(n>=1) {
			//System.out.print(n+" ");
			print(n-1);
			System.out.print(n+" ");
		}
	}
}

public class PrintNaturalNatural {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		int n = obj.nextInt();
		
		Demo1.print(n);
	}
	
}
