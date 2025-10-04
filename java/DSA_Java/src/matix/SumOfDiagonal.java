package matix;

import java.util.Scanner;

public class SumOfDiagonal {

	public static void main(String[] args) {
		Scanner obj = new Scanner(System.in);

		System.out.println("Enter row value: ");
		int row = obj.nextInt();
		
		System.out.println("Enter col value: ");
		int col = obj.nextInt();
		
		int a[][] = new int[row][col];
		
		int i,j;
		
		System.out.println("Enter matrix element : ");
		for(i=0; i<row; i++) {
			for(j=0; j<col; j++) {
				a[i][j] = obj.nextInt();
			}
		}
		
		System.out.println("Matrix : ");
		for(i=0; i<row; i++) {
			for(j=0; j<col; j++) {
				System.out.print(a[i][j]+" ");
			}
			System.out.println();
		}
		
		int sum = 0;
		

		for(i=0; i<row; i++) {
			for(j=0; j<col; j++) {
				if(i == j) {
					sum = sum + a[i][j];
				}
			}
		}
		
		System.out.println("Some of Daigonal : "+ sum);
	}
	
}
