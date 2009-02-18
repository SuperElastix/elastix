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
  Date:      $Date: 2008-10-02 15:23:50 +0200 (Thu, 02 Oct 2008) $
  Version:   $Revision: 2255 $

  Copyright (c) 2002 Insight Consortium. All rights reserved.
  See ITKCopyright.txt or http://www.itk.org/HTML/Copyright.htm for details.

     This software is distributed WITHOUT ANY WARRANTY; without even 
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR 
     PURPOSE.  See the above copyright notices for more information.

=========================================================================*/
#include <iostream>
#include <string>
//#include <stdlib.h>
#include <vector>
#include "itk_hash_map.h"
#include "itkMacro.h"

// needed for gcc4.3:
#include <cstring>

#ifndef __VISParameterFile_h__
#define __VISParameterFile_h__
namespace VISParameterFile
{

inline void DIE(const char *s)
{
  itkGenericExceptionMacro(<< "VPF fatal error: " << s );
  //std::cerr << "VPF fatal error: " << s << std::endl;
  //exit(-1);
}
  
typedef enum { VALID, INVALID } ReturnStatusType;
   
class Exception
{
public:
  Exception(std::string s) : info(s) {}
  Exception() { info = "no information"; }
  std::string info;
};

inline std::ostream& operator<<(std::ostream& s, const Exception &f)
{ s << "VPF::Exception: " << f.info; return s; }

//typedef itk::hash<const char *> strhash;
class strhash : public itk::hash<const char *>
{
public:
  itk::hash<const char *> hasher;
  ::size_t operator()(const std::string &s) const
  {
    //    return itk::hash<const char *>::operator()(s.c_str());
    return hasher(s.c_str());
  }
};
  
struct eqstr
{
  bool operator()(const std::string& s1, const std::string& s2) const
  {
    return s1==s2;
  }
};
  
typedef enum{
  VPF_DECIMAL=0, VPF_INTEGER=1, VPF_STRING=2, VPF_UNKNOWN=3 } value_type;

class GenericValue
{
public:
  GenericValue(bool b) {m_Valid = b;}
  GenericValue() : m_Valid(true) {}
	virtual ~GenericValue() {}
  virtual value_type GetValueType() const    { return VPF_UNKNOWN; }
  virtual void PrintSelf(std::ostream &s) const { s << "VPF_UNKNOWN_VALUE"; }
  bool valid() const { return m_Valid; }
  void valid(bool b) { m_Valid = b; }
  bool m_Valid;
};

template<class T> struct ValueTraits
{
  static const value_type ValueType;
  static void PrintValue(std::ostream &s, const T &v) { s << v; }
};
template<> struct ValueTraits<float>
{
  static const value_type ValueType;
  static void PrintValue(std::ostream &s, const float &v) { s << v; }
};
template<> struct ValueTraits<long int>
{
  static const value_type ValueType;
  static void PrintValue(std::ostream &s, const long int &v){ s << v; }
};
template<> struct ValueTraits<std::string>
{
  static const value_type ValueType;
  static void PrintValue(std::ostream &s, const std::string &v)
    { s << "\"" << v << "\""; }
};

template<class T>
class Value : public GenericValue
{
  T m_Value;
public:
  T GetValue() const { return m_Value; }
  T getValue() const { return m_Value; }
  void SetValue( const T& v ) { m_Value = v; }
  virtual value_type GetValueType() const { return ValueTraits<T>::ValueType; }
  virtual void PrintSelf(std::ostream &s) const
  { ValueTraits<T>::PrintValue(s, m_Value); }
  virtual ~Value() {}
};

class ValueFactory
{
public:
  static GenericValue *Create( const value_type );
  static GenericValue *Copy( GenericValue *v);
  static void Delete( GenericValue *v );
};

/** \class Parameter
 * \brief Holds a lists of generic values and a keyword associated with those
 * values.
 *
 * Parameter is implemented as an array of pointers to the value superclass
 * because each value element's specific type is unknown, but must be
 * preserved.
 */
class Parameter : public std::vector<GenericValue *>
{
  bool m_Valid;
  std::string m_Key;
  static GenericValue m_InvalidValue;

  void PushBack( GenericValue *v )  // Private versions that use pointers.
  {                                 // Necessary for object copying because
    this->push_back( ValueFactory::Copy( v ) ); // const references cannot
  }                                             // be dynamically downcast.
  void PushFront( GenericValue *v )
  {
    this->insert( this->begin(),  ValueFactory::Copy( v ) );
  }
  
public:
  typedef std::vector<GenericValue *> Superclass;
  Parameter(bool b) { m_Valid = b; }
  Parameter() : m_Valid(true) { }
  ~Parameter() { this->Clear(); }
  const Parameter& operator=(const Parameter &p)
  {
    this->Clear();  m_Valid = p.m_Valid;  m_Key = p.m_Key;
    for (const_iterator it = p.begin(); it < p.end(); ++it)
      { this->PushBack(*it); }
    return *this;
  }
  Parameter(const Parameter& p): Superclass() { *this = p; }
  GenericValue *operator[]( std::vector<GenericValue *>::size_type n) const
    {
      if (this->valid() == false) return &m_InvalidValue;
      else return this->GetElement(n);
    }  
  GenericValue *GetElement( std::vector<GenericValue *>::size_type n) const ;
  void Clear()
  {
    for (iterator it = this->begin(); it < this->end(); ++it)
      { ValueFactory::Delete(*it); }
    this->clear();
  }  
  void PushBack( GenericValue &v )  // Passing by reference allows us
  {                                 // to keep the subtype info
    this->push_back( ValueFactory::Copy( &v ) );
  }
  void PushFront( GenericValue &v )
  {
    this->insert( this->begin(),  ValueFactory::Copy( &v ) );
  }
  bool valid() const { return m_Valid; }
  void valid(const bool b) { m_Valid = b; }
  void Key(const std::string &k) { m_Key = k; }
  const char *getName() const { return m_Key.c_str();}
  std::string Key() const { return m_Key; };
  void PrintSelf(std::ostream &s) const
  {
    s << "( " << this->Key() << " ";
    for (const_iterator it = this->begin(); it < this->end(); ++it)
      { (*it)->PrintSelf(s); s << " ";}
    s << ")" << std::endl;
  }
};

typedef itk::hash_map<std::string, Parameter, strhash, eqstr> File;

class ParameterFile
{
  static Parameter m_InvalidParameter;
  File m_File;
public:
  ParameterFile() {}
  ParameterFile(const char *s)        { this->Initialize(s);         }
  ParameterFile(const std::string &s) { this->Initialize(s.c_str()); }
  void PrintSelf(std::ostream &s) const
  {
    for (itk::hash_map<std::string, Parameter, strhash, eqstr>::const_iterator
           it = m_File.begin(); it != m_File.end(); ++it )
      { (*it).second.PrintSelf(s); }
  }
  void Initialize(const char *);
  bool valid()    const { return m_File.empty(); }
  ::size_t size() const {  return m_File.size(); }
  bool empty()    const { return m_File.empty(); }
  const Parameter &operator[](const std::string &s);
  const Parameter &operator[](const char *s)
     {  std::string temp = s; return this->operator[](temp);  }
};

inline ReturnStatusType set(long &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;

  if (value->GetValueType() != ValueTraits<long>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //  throw Exception("VPF::set: Operand is of the wrong type");
    }
  operand = dynamic_cast<Value<long> *>(value)->GetValue();
  return VALID;
}

inline ReturnStatusType set(unsigned long &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;
  
  if (value->GetValueType() != ValueTraits<long>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //    throw Exception("VPF::set: Operand is of the wrong type");
    }

  operand = dynamic_cast<Value<long> *>(value)->GetValue();
  return VALID;
}

inline ReturnStatusType set(short &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;
  
  if (value->GetValueType() != ValueTraits<long>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //  throw Exception("VPF::set: Operand is of the wrong type");
    }

  operand = dynamic_cast<Value<long> *>(value)->GetValue();
  return VALID;
}

inline ReturnStatusType set(unsigned short &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;

  if (value->GetValueType() != ValueTraits<long>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //  throw Exception("VPF::set: Operand is of the wrong type");
    }

  operand = dynamic_cast<Value<long> *>(value)->GetValue();
  return VALID;
}

inline ReturnStatusType set(int &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;

  if (value->GetValueType() != ValueTraits<long int>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //        throw Exception("VPF::set: Operand is of the wrong type");
    }

  operand = dynamic_cast<Value<long int> *>(value)->GetValue();
  return VALID;
}

inline ReturnStatusType set(unsigned int &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;

  if (value->GetValueType() != ValueTraits<long int>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //        throw Exception("VPF::set: Operand is of the wrong type");
    }

  operand = dynamic_cast<Value<long int> *>(value)->GetValue();
  return VALID;
}

inline ReturnStatusType set(float &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;

  if (value->GetValueType() != ValueTraits<float>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //      throw Exception("VPF::set: Operand is of the wrong type");
    }

  operand = dynamic_cast<Value<float> *>(value)->GetValue();
  return VALID;
}

// Remove support for char *, use std::string instead.
/*inline ReturnStatusType set(char *operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;

  if (value->GetValueType() != ValueTraits<std::string>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //  throw Exception("VPF::set: Operand is of the wrong type");
    }

  // strcpy was deprecated in VS2008: so I out-commented next line
  // and replaced it with the following two lines.
  strcpy(operand, dynamic_cast<Value<std::string> *>(value)->GetValue().c_str());
  // However, this does not exist in VS2003
  //const char * tmpchar( dynamic_cast<Value<std::string> *>(value)->GetValue().c_str() );
  //strcpy_s( operand, strlen( tmpchar ), tmpchar );  
  return VALID;
}*/

inline ReturnStatusType set(std::string &operand, GenericValue *value)
{
  if (value->valid() == false) return INVALID;

  if (value->GetValueType() != ValueTraits<std::string>::ValueType)
    {
      DIE("VPF::set: Operand is of the wrong type");
      //      throw Exception("VPF::set: Operand is of the wrong type");
    }

  operand = dynamic_cast<Value<std::string> *>(value)->GetValue();
  return VALID;
}

template <class T>
inline ReturnStatusType set(GenericValue *value, T &operand)
  { return set(operand, value); }

inline std::ostream& operator<<(std::ostream& s, const ParameterFile &f)
 { f.PrintSelf(s); return s; }

} // end namespace VISParameterFile

namespace VPF = VISParameterFile;

#endif
