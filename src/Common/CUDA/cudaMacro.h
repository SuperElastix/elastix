/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __cudaMacro_h
#define __cudaMacro_h

/* cuda version of some of the itk macros */
namespace cuda
{

#define DBG_FUNC(NAME, PARAMETERS_DECLR, PARAMETERS_CALL) \
  inline  cudaError_t NAME PARAMETERS_DECLR { \
  cudaError_t err = ::NAME PARAMETERS_CALL;   \
  cudaCheckMsg(#NAME" failed!");             \
  return err;                                  \
}

#define cudaGetConstMacro(name,type) \
  virtual type Get##name () const {  \
  return this->m_##name;             \
}

#define cudaSetMacro(name,type)                      \
  virtual void Set##name (const type _arg) {         \
  if (this->m_##name != _arg) this->m_##name = _arg; \
}

#define cudaGetMacro(name,type) \
  virtual type Get##name () {   \
  return this->m_##name;        \
}

#define cudaBooleanMacro(name) \
  virtual void name##On() {    \
  this->Set##name(true);       \
}                              \
  virtual void name##Off() {   \
  this->Set##name(false);      \
}

}; /* namespace cuda */

#endif // end #ifndef __cudaMacro_h

