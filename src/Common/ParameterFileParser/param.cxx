/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/

/** This file is copied from the ITK, and slightly modified.
 * The original copyright message is pasted here: */
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
#if defined(_MSC_VER)
#pragma warning ( disable : 4786 )
#endif

#include "param.h"
extern "C"{
#include <stdio.h>
}
extern FILE *yyin;
extern int yyparse();
extern void yyrestart(FILE *);
extern void reset_yyvalues();
extern int syntax_error_count;
extern VISParameterFile::File VPF_file_parse_result;

namespace VISParameterFile {
  
const value_type ValueTraits<float>::ValueType       = VPF_DECIMAL;
const value_type ValueTraits<long int>::ValueType    = VPF_INTEGER;
const value_type ValueTraits<std::string>::ValueType = VPF_STRING;

GenericValue Parameter::m_InvalidValue = GenericValue(false);
Parameter ParameterFile::m_InvalidParameter = Parameter(false);

GenericValue *ValueFactory::Create( const value_type t)
  {
    if (t==VPF_DECIMAL) return new Value<float>();
    if (t==VPF_INTEGER) return new Value<long int>();
    if (t==VPF_STRING)  return new Value<std::string>();
    else return 0;
  }

GenericValue *ValueFactory::Copy( GenericValue *v)
{
  GenericValue *temp = Create( v->GetValueType() );
  switch (v->GetValueType()) {
  case VPF_DECIMAL:
    *dynamic_cast<Value<float> *>(temp)
      = *dynamic_cast<Value<float> *>(v);
    break;
  case VPF_INTEGER:
    *dynamic_cast<Value<long int> *>(temp)
      = *dynamic_cast<Value<long int> *>(v);
    break;
  case VPF_STRING:
    *dynamic_cast<Value<std::string> *>(temp)
      = *dynamic_cast<Value<std::string> *>(v);
    break;
  default:
    return 0;
  }
  return temp;
}

void ValueFactory::Delete( GenericValue *v)
{
  switch (v->GetValueType()) {
  case VPF_DECIMAL:
    delete dynamic_cast<Value<float> *>(v);
    break;
  case VPF_INTEGER:
    delete dynamic_cast<Value<long int> *>(v);
    break;
  case VPF_STRING:
    delete dynamic_cast<Value<std::string> *>(v);
    break;
  default:
    break;
  }
}

const Parameter& ParameterFile::operator[](const std::string &s)
{
  if ( this->m_File.find(s) == this->m_File.end() )
    {
      return this->m_InvalidParameter;
      //      std::string temp = "ParameterFile::operator[]: Parameter \""
      //        + s + "\" was not found.";
      //      throw Exception(temp);
      //      DIE(temp.c_str());
    }
  return this->m_File[s];
}

GenericValue * Parameter
::GetElement( std::vector<GenericValue *>::size_type n) const
{
  typedef std::vector<GenericValue *> Superclass;
  if (this->valid() == false)
    {
      return &this->m_InvalidValue;
      //      DIE("VPF::Parameter: request for element in an invalid parameter");
    }
  if (this->size() <= n)
    {
      //      throw Exception(
      //      "VPF::Parameter: request for element past the end of parameter \""
      //      + this->Key()+"\"" );
      return &this->m_InvalidValue;
      //      DIE("VPF::Parameter: request for element past the end of parameter");
    }
  return Superclass::operator[](n);
}


void ParameterFile::Initialize( const char* fn )
{
  if ((::yyin = ::fopen(fn, "r"))==NULL)
  //FILE * yyin;
  //if ( ::fopen_s( &yyin, fn, "r" ) != 0 )
    {
      std::string temp = "ParameterFile::Could not open input file \""
      + std::string(fn) + "\".";
      //      throw Exception(temp);
      DIE( temp.c_str() );
    }
  else
    {
      ::yyrestart( yyin );
      ::reset_yyvalues();
      ::yyparse();
      ::fclose( yyin );
      
      if (::syntax_error_count > 0)
        {
          std::string temp =  "ParameterFile::Input file \""
            + std::string(fn) + "\" contains syntax errors. ";
          //          throw Exception(temp);
          DIE(temp.c_str());
        }
      else
        {
          this->m_File = ::VPF_file_parse_result;
        }
    }
  
}
  
  
}
