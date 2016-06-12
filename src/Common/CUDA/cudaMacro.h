/*=========================================================================
 *
 *  Copyright UMC Utrecht and contributors
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *        http://www.apache.org/licenses/LICENSE-2.0.txt
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 *
 *=========================================================================*/
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

