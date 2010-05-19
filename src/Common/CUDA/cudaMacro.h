/* $Id$ */
#pragma once

/* cuda version of some of the itk macros */
namespace cuda
{
	#define DBG_FUNC(NAME, PARAMETERS_DECLR, PARAMETERS_CALL) \
	  inline  cudaError_t NAME##PARAMETERS_DECLR {            \
	    cudaError_t err = ::NAME##PARAMETERS_CALL;            \
	    cudaCheckMsg(#NAME##" failed!");                      \
	    return err;                                           \
	  }

	#define cudaGetConstMacro(name,type) \
	  virtual type Get##name () const {  \
	    return this->m_##name;           \
	  }

	#define cudaSetMacro(name,type)                        \
	  virtual void Set##name (const type _arg) {           \
	    if (this->m_##name != _arg) this->m_##name = _arg; \
	  }

	#define cudaGetMacro(name,type) \
	  virtual type Get##name () {   \
	    return this->m_##name;      \
	  }
}; /* namespace cuda */
