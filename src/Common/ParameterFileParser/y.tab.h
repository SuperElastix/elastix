/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date$
  Version:   $Revision$

  Copyright (c) 2002 Insight Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
typedef union
{
  long int integer;
  float decimal;
  std::string *string;
  Parameter*  Parameter_ptr;
  File*       File_ptr;
  GenericValue * GenericValue_ptr;
  std::vector<GenericValue *> * ValueArray_ptr;
} YYSTYPE;

#define NUM 257
#define DEC_NUM 258
#define IDENT 259
#define STRING  260
#define EQUALS  261
#define LPAREN  262
#define RPAREN  263
#define NEWLINE 264

extern YYSTYPE yylval;
