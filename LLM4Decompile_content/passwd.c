#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PASSWORD_LENGTH 12

void generate_password(char *password, size_t length) {
        const char *charset = "abcdefghijklmnopqrstuvwxyz"
                              "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
                              "0123456789"
                              "!@#$%^&*()_+-=[]{}|;:,.<>?";
            
        if (length < 1) return;

        srand((unsigned int)time(NULL)); // Seed the random number generator
                                                 
        for (size_t i = 0; i < length; i++) {
            size_t index = rand() % (sizeof(charset) - 1); // Random index
            password[i] = charset[index]; // Assign random character
        }
                                                                              
             password[length] = '\0'; // Null-terminate the password string
}
                                                 
int main() {
    char password[PASSWORD_LENGTH + 1]; // +1 for the null terminator
    generate_password(password, PASSWORD_LENGTH);
    printf("Generated Password: %s\n", password);
    return 0;
}
                                                 
