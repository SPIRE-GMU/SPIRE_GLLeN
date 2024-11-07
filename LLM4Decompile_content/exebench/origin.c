#include <stdint.h>
#include <stdio.h>
#include <sys/wait.h>

#include <stdio.h>

#include <stdlib.h>

#include <unistd.h>

#include <string.h>

# 1 
int main(int argc, char* argv[])
{
 int row_number = 1;
 int pipe_dsc_back [2];
 int pipe_dsc_forth [2];
 int buf_len = 0;
 char buf [27];
 char state;


 if (argc < 2) {
  syserr("Too few arguments\n");
 }


 if((row_number = atoi(argv[1])) <= 0) {
  syserr("You have to give number greater than 0\n");
 }


 if(pipe(pipe_dsc_back) == -1) {
  syserr("Error in creating pipe_forth\n");
 }
 if(pipe(pipe_dsc_forth) == -1)
  syserr("Error in creating pipe_dsc_forth\n");
 switch(fork()) {
  case -1:
   syserr("Error in fork\n");

  case 0:

   if(dup2(pipe_dsc_back[1], STDOUT_FILENO) == -1) {
    syserr("P:Error in dup pipe_dsc_back[1]\n");
   }
   if(dup2(pipe_dsc_forth[0], STDIN_FILENO) == -1) {
    syserr("P:Error in dup pipe_dsc_forth[0]\n");
   }


   if(close(pipe_dsc_back[0]) == -1) {
    syserr("P:Error in close pipe_dsc_back[0]\n");
   }
   if(close(pipe_dsc_back[1]) == -1) {
    syserr("P:Error in close pipe_dsc_back[1]\n");
   }
   if(close(pipe_dsc_forth[0]) == -1) {
    syserr("P:Error in close pipe_dsc_forth[0]\n");
   }
   if(close(pipe_dsc_forth[1]) == -1) {
    syserr("P:Error in close pipe_dsc_forth[1]\n");
   }

   execl("./w", "w", (char *) 0);
   syserr("Error in execl(w).\n");

  default:
   if(close(pipe_dsc_back[1]) == -1) {
    syserr("P:Error in close pipe_dsc_back[1]\n");
   }
   if(close(pipe_dsc_forth[0]) == -1) {
    syserr("P:Error in close pipe_dsc_forth[0]\n");
   }
 }

 state = (row_number == 1) ? 'q' : 'a';

 int i = 1;
 int first_value = 0;
 while (i <= row_number) {


  memset(buf, 0, 27);
  if (i == (row_number - 1)) {
   state = 'f';
  }
  buf_len = sprintf(buf, "%d %d", state, first_value);
  if ((buf_len = write (pipe_dsc_forth[1], buf, sizeof(buf))) == -1)
   syserr("Pascal: iteration nr %d. Error in write", i);
  if (i < row_number)
   if ((buf_len = read (pipe_dsc_back[0], buf, sizeof(buf))) == -1)
    syserr("Pascal. Error in read\n");
  i++;

  if (i == row_number)
   state = 'q';
 }

 i = 1;

 while (i <= row_number) {
  if ((buf_len = read (pipe_dsc_back[0], buf, sizeof(buf))) == -1)
   syserr("Pascal. Error in read\n");
  printf("%ld ", strtol(buf+3, NULL, 10) );
  i++;
 }
 printf("\n");


 if(wait(0) == -1) {
  syserr("Error in wait\n");
 }

 return 0;

}