package matix;

import java.util.Scanner;

public class MatrixMultiplication {

	public static void main(String[] args) {
		
		Scanner obj = new Scanner(System.in);
		
		System.out.println("Enter matrix 1 row1 size : ");
		int rsize1 = obj.nextInt();
		
		System.out.println("Enter matrix 1 column1 size : ");
		int csize1 = obj.nextInt();
		
		System.out.println("Enter matrix 2 row2 size : ");
		int rsize2 = obj.nextInt();
		
		System.out.println("Enter matrix 2 column2 size : ");
		int csize2 = obj.nextInt();
		
		if(rsize1 == csize2) {
			int a[][] = new int[rsize1][csize1];
			int b[][] = new int[rsize2][csize2];
			int c[][] = new int[rsize2][csize2];
			System.out.println("Enter matrix 1 element "+ (rsize1*csize1) +" one-by-one");
			
			for(int i=0;i<rsize1;i++) {
				for(int j=0; j<csize1;j++) {
					a[i][j] = obj.nextInt();
				}
			}
			
			System.out.println("Enter matrix 2 element "+ (rsize2*csize2) +" one-by-one");
			for(int i=0;i<rsize2;i++) {
				for(int j=0; j<csize2;j++) {
					b[i][j] = obj.nextInt();
				}
			}
			
			for(int i=0; i<rsize1; i++) {
				for(int j=0; j<csize2; j++) {
					c[i][j] = 0;
					for(int k=0; k<csize1; k++) {
						c[i][j] = c[i][j] + (a[i][k]*b[k][j]);
					}
				}
			}
			
			System.out.println("Matrix elements are: ");
			for(int i=0; i<rsize1; i++) {
				for(int j=0; j<csize1; j++) {
					System.out.print(a[i][j]+"["+i+","+j+"]"+" ");
				}
				System.out.println();
			}
			
			System.out.println("Matrix elements are: ");
			for(int i=0; i<rsize1; i++) {
				for(int j=0; j<csize1; j++) {
					System.out.print(b[i][j]+"["+i+","+j+"]"+" ");
				}
				System.out.println();
			}
			
			System.out.println("Matrix elements are: ");
			for(int i=0; i<rsize1; i++) {
				for(int j=0; j<csize1; j++) {
					System.out.print(c[i][j]+"["+i+","+j+"]"+" ");
				}
				System.out.println();
			}
		} else {
			System.out.println("Matrix can't be Multiple...");
		}
		
		
	}
	
}
