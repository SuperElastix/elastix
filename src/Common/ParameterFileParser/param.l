/*=========================================================================

  Program:   Insight Segmentation & Registration Toolkit
  Module:    $RCSfile$
  Language:  C++
  Date:      $Date: 2008-06-20 11:27:23 +0200 (Fri, 20 Jun 2008) $
  Version:   $Revision: 1707 $

  Copyright (c) 2002 Insight Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
%{

#include <iostream>
#include<string>
extern "C" {
#include <stdlib.h>
}
#include "param.h"
 using namespace VISParameterFile;
#include "y.tab.h"

#define MAXERRORS 	20
#define ERRSTRING 	"unrecognized word"
#define BADTOKEN	-1
#define TABSPACE	8
 
// The following three macros are used by yylex() to manage line and
// column numbers.  INCR_COL updates the column number for all non-error
// matches except for tabs (handled in the TAB rule).  INCR_LINE updates
// the current line number.  INCR_TOKEN updates the column number in the
// error state.

#define INCR_COL	token_tail=token_head; token_head+=yyleng
#define INCR_LINE	++line_no; token_tail=token_head=1;
#define INCR_TOKEN	token_tail=token_head; token_head+=err_token_size;

  //#define NUM 1
  //#define DEC_NUM 2
  //#define IDENT 3
  //#define EQUALS 4

        // Global variables for managing current line number, column, and error
        // count.

  int line_no=1, token_head=1, token_tail=1, error_count=0, err_token_size=0;


        // DEFINITIONS:
        //
        // DELIM        Since tabs and newlines must be handled differently,
        //              delimeters are just spaces.
        // WS           Whitespace.  One or more delimeters.
        // DIGIT        A single digit.
        // LETTER       A single upper or lower-case letter.
        // ID           Identifiers are a letter followed by any number of
        //              letters or numbers or underscores.
        // COMMENT      A comment is two forward slashes followed by any string
        //              of characters up to a newline.
        // TAB          A single tab character.
        // NEWLINE      A single newline character.
  
%}

%option noyywrap

STRING \"[^\n"]+\"
NEWLINE	\n
EQUALS \=
LPAREN \(
RPAREN \)
DELIM	[ ]
WS	{DELIM}+
DIGIT	[0-9]
LETTER	[a-zA-Z]
ID	{LETTER}({LETTER}|{DIGIT}|_)*
NUMBER	 {DIGIT}+|\-{DIGIT}+
DEC_NUMBER {DIGIT}*\.{DIGIT}+|\-{DIGIT}*\.{DIGIT}+ 
COMMENT	\/\/.*
TAB	\t


%x	ERROR

%%
{NEWLINE} { INCR_LINE; }

{TAB} { /* Increment the column number to align with tab stops. */
  token_tail=token_head; 
  token_head+=TABSPACE-((token_head-1)%TABSPACE);
}

{WS} { INCR_COL; }

{COMMENT} { INCR_LINE;  /* Comments: do nothing. */	}

{STRING} {
  INCR_COL;
  //  yylval.string = new char[yyleng-2];
  //  yytext[yyleng-1]='\0'; /* Strip of trailing quote character */
  //  strcpy(yylval.string, yytext+1);  /* Strip leading quote character */
  yylval.string = new std::string();
  *yylval.string = yytext;
  yylval.string->erase(yylval.string->begin()); // strip leading quote
  yylval.string->erase(yylval.string->end() -1); // strip trailing quote
  return STRING;
}


{ID} {
  INCR_COL;
  //  yylval.string = new char[yyleng];
  //  strcpy(yylval.string, yytext);
  yylval.string = new std::string();
  *yylval.string = yytext;
  return IDENT;
}


{NUMBER} {
  INCR_COL;
  yylval.integer=atol(yytext);
  return NUM;
}

{EQUALS} {
  INCR_COL;
  return EQUALS;
}

{LPAREN} {
  INCR_COL;
  return LPAREN;
}
{RPAREN} {
  INCR_COL;
  return RPAREN;
}


{DEC_NUMBER} {
  INCR_COL;
  yylval.decimal=atof(yytext);
  return DEC_NUM;
}

.|{NUMBER}{ID} {
  err_token_size=yyleng;  /* This rule catches errors. Anything */
  error_count++;	      /* not defined above, or a number     */
  BEGIN ERROR;		      /* concatenated with an identifier is */
  yymore();		          /* an error.  Go to the error state.  */
  
}


<ERROR>{NEWLINE}|{WS}|{TAB} {
  /* This rule finds the end of an   */
  /* error token.  Whatever was      */
  /* appended to yytext by this rule */
  /* is thrown back by yyless and    */
  /* we return to the initial state. */
  
  yyless(err_token_size);
  INCR_TOKEN;
  BEGIN INITIAL; 
  std::cout << "yylex(): Line "<< line_no << ", " <<  ERRSTRING << "  "
     << yytext << std::endl;
  if (error_count > MAXERRORS)
  {
    std::cout << "Too many lexical errors ("
       << error_count
       <<"), quitting.\n";
    exit(1);
  }
}

<ERROR>{ID} {
  /* This rule and the next add characters */
  /* up to a whitespace, tab, newline, or  */
  /* operator onto the token.  The ID rule */
  /* is here for speed and to prevent      */
  /* crashes at end-of-file.               */
  
  err_token_size=yyleng;
  yymore();
}

<ERROR>. {
  err_token_size=yyleng;
  yymore();
}

%%
