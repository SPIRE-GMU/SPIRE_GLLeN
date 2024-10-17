void generate_password(char* password, size_t length)
{
    const char charset[] = "0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ";
    const size_t charset_size = sizeof(charset) - 1;

    if (length > 0)
    {
        srand((unsigned)time(NULL));

        for (size_t i = 0; i < length; ++i)
        {
            int r = rand();
            size_t index = (size_t)(r % charset_size);
            password[i] = charset[index];
        }

        password[length] = '\0';
    }
}