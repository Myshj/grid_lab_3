#include <mpi/mpi.h>

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "linalg.h"

/* Ім'я вхідного файлу */
const char *input_file_MA = "matrix.txt";
const char *input_file_b = "b.txt";
const char* output_file_x = "out_vector_x";

/* Тег повідомленя, що містить стовпець матриці */
//const int COLUMN_TAG = 0x1;

/* Основна функція (програма обчислення визначника) */
int main(int argc, char *argv[])
{
    /* Ініціалізація MPI */
    MPI_Init(&argc, &argv);

    /* Отримання загальної кількості задач та рангу поточної задачі */
    int np, rank;
    MPI_Comm_size(MPI_COMM_WORLD, &np);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    /* Зчитування даних в задачі 0 */
    struct my_matrix *input_matrix;
    int matrix_size;
    if(rank == 0)
    {
        input_matrix = read_matrix(input_file_MA);

        if(input_matrix->rows != input_matrix->cols) {
            fatal_error("Matrix is not square!", 4);
        }
        matrix_size = input_matrix->rows;
    }


    /* Розсилка всім задачам розмірності матриць та векторів */
    MPI_Bcast(&matrix_size, 1, MPI_INT, 0, MPI_COMM_WORLD);

    /*Рассылка всем задачам вектора свободных членов.*/
    struct my_vector* b = vector_alloc(matrix_size, 0.);
    if(rank == 0){
        b = read_vector(input_file_b);
    }
    MPI_Bcast(
            b->data, matrix_size, MPI_DOUBLE, 0, MPI_COMM_WORLD
    );

    /* Выделяем память для строк в локальной памяти процессов. */
    struct my_vector *local_row = vector_alloc(matrix_size, .0);
    struct my_vector* local_l = vector_alloc(matrix_size, 0.);
    struct my_vector* temp_row = vector_alloc(matrix_size, 0.);

    /* Розсилка рядків матриці з задачі 0 в інші задачі */
    if(rank == 0)
    {
        MPI_Scatter(
                input_matrix->data, matrix_size, MPI_DOUBLE,
                local_row->data, matrix_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD
        );

        free(input_matrix);
    }
    else
    {
        MPI_Scatter(
                NULL, 0, MPI_DATATYPE_NULL,
                local_row->data, matrix_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD
        );
    }

    int current_leader = 0;
    while (current_leader < matrix_size){
        if (current_leader == rank){
            for (int i = 0; i < matrix_size; i++){
                temp_row->data[i] = local_row->data[i];
            }

//                for (int i = 0; i < matrix_size; i++){
//                    printf("%f\t", temp_row->data[i]);
//                }
//                printf("\n");

        }

        MPI_Bcast(
                temp_row->data, matrix_size, MPI_DOUBLE, current_leader, MPI_COMM_WORLD
        );

        if(rank > current_leader){
            local_l->data[current_leader] = local_row->data[current_leader] / temp_row->data[current_leader];
            for (int j = 0; j < matrix_size; j++){
                local_row->data[j] -= local_l->data[current_leader] * temp_row->data[j];
            }
        }
        current_leader++;
    }

    ///////////////////////////////////////////////////////////////////////////////////////////////////////////

    struct my_matrix* global_l;
    if(rank == 0){
        global_l = matrix_alloc(matrix_size, matrix_size, 0.);
        MPI_Gather(
                local_l->data, matrix_size, MPI_DOUBLE,
                global_l->data, matrix_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD
        );
    }else{
        MPI_Gather(
                local_l->data, matrix_size, MPI_DOUBLE,
                NULL, 0, MPI_DOUBLE,
                0, MPI_COMM_WORLD
        );
    }

    struct my_matrix* global_matrix;
    if(rank == 0){
        global_matrix = matrix_alloc(matrix_size, matrix_size, 0.);
        MPI_Gather(
                local_row->data, matrix_size, MPI_DOUBLE,
                global_matrix->data, matrix_size, MPI_DOUBLE,
                0, MPI_COMM_WORLD
        );
    }else{
        MPI_Gather(
                local_row->data, matrix_size, MPI_DOUBLE,
                NULL, 0, MPI_DOUBLE,
                0, MPI_COMM_WORLD
        );
    }

//    if(rank == 0){
//        for (int i = 0; i < matrix_size; i++){
//            for (int j = 0; j < matrix_size; j++){
//                printf("%f\t", global_matrix->data[i * matrix_size + j]);
//            }
//            printf("\n");
//        }
//    }

    if(rank == 0) {
        struct my_vector *y = vector_alloc(matrix_size, 0.);
        for (int i = 0; i < matrix_size; i++){
            y->data[i] = b->data[i];
            for(int s = 0; s < i; s++){
                y->data[i] -= y->data[s] * global_l->data[i * matrix_size + s];
            }
        }
        struct my_vector* x = vector_alloc(matrix_size, 0.);
        for(int i = matrix_size - 1; i >= 0; i--){
            x->data[i] = y->data[i];
            for (int s = i + 1; s < matrix_size; s++){
                x->data[i] -= x->data[s] * global_matrix->data[i * matrix_size + s];
            }
            x->data[i] /= global_matrix->data[i * matrix_size + i];
        }
        write_vector(output_file_x, x);
    }


    /* Повернення виділених ресурсів */
    return MPI_Finalize();
}

