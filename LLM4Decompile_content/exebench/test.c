void main()
{}
static void num2str(char *c, int i)
{
 c[0] = i / 10 + '0';
 c[1] = i % 10 + '0';
}