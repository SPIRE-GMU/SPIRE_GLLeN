#include <stdint.h>
None
#define NULL ((void*)0)
typedef unsigned long size_t;  // Customize by platform.
typedef long intptr_t; typedef unsigned long uintptr_t;
typedef long scalar_t__;  // Either arithmetic or pointer type.
/* By default, we understand bool (as a convenience). */
typedef int bool;
#define false 0
#define true 1

/* Forward declarations */

/* Type definitions */
struct S_STRING_STRUCT {int size; int capacity; int /*<<< orphan*/  string; } ;
typedef  int /*<<< orphan*/ * s_str ;

/* Variables and functions */
 int /*<<< orphan*/ * malloc (int) ; 
 int /*<<< orphan*/  memcpy (int /*<<< orphan*/ ,int /*<<< orphan*/ ,int) ; 
 int /*<<< orphan*/ * s_str_create () ; 


void main()
{}
s_str s_str_create_from_s_str(const s_str *const s_str_ptr_for_create){
    s_str new_str;
    if(s_str_ptr_for_create == NULL){
        new_str = s_str_create();
    }else{
        new_str = malloc(sizeof(struct S_STRING_STRUCT) + sizeof(char) * (*(struct S_STRING_STRUCT *) *s_str_ptr_for_create).size);
        if(new_str != NULL){
            (*(struct S_STRING_STRUCT *) new_str).capacity = (*(struct S_STRING_STRUCT *) *s_str_ptr_for_create).size;
            (*(struct S_STRING_STRUCT *) new_str).size = (*(struct S_STRING_STRUCT *) *s_str_ptr_for_create).size;
            memcpy((*(struct S_STRING_STRUCT *) new_str).string, (*(struct S_STRING_STRUCT *) *s_str_ptr_for_create).string, sizeof(char) * (*(struct S_STRING_STRUCT *) *s_str_ptr_for_create).size);
        }
    }
    return new_str;
}
