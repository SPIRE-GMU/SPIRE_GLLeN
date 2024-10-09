#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define PASSWORD_LENGTH 12

void generate_password(char* password, size_t length)
{
    const char* charset = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789!\"%&'()*+,-./:;<=>?@[\\]^_`";
    size_t charset_size = sizeof(charset) - 1;

    if (length > 0)
    {
        srand(time(NULL));

        for (size_t i = 0; i < length; ++i)
        {
            int r = rand();
            size_t index = (size_t)(r % charset_size);
            password[i] = charset[index];
        }

        password[length] = '\0';
    }
}
int main() {
    char password[PASSWORD_LENGTH + 1]; // +1 for the null terminator
    generate_password(password, PASSWORD_LENGTH);
    printf("Generated Password: %s\n", password);
    return 0;
}

