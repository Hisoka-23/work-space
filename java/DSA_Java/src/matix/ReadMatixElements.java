package matix;

import java.util.Scanner;

public class ReadMatixElements {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter matrix row size : ");
		int rsize = obj.nextInt();
		
		System.out.println("Enter matrix column size : ");
		int csize = obj.nextInt();
		
		int a[][] = new int[rsize][csize];
		
		System.out.println("Enter matrix element "+ (rsize*csize) +" one-by-one");
		
		for(int i=0;i<rsize;i++) {
			for(int j=0; j<csize;j++) {
				a[i][j] = obj.nextInt();
			}
		}
		
		System.out.println("Matrix elements are: ");
		for(int i=0; i<rsize; i++) {
			for(int j=0; j<csize; j++) {
				System.out.print(a[i][j]+"["+i+","+j+"]"+" ");
			}
			System.out.println();
		}
	}
	
}
