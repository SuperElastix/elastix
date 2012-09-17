/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUTranslationTransformBase_h
#define __itkGPUTranslationTransformBase_h

#include "itkGPUTransformBase.h"

namespace itk
{
/** Create a helper GPU Kernel class for itkGPUTranslationTransformBase */
itkGPUKernelClassMacro( GPUTranslationTransformBaseKernel );

/** \class GPUTranslationTransformBase
*/
template<
  class TScalarType = float,   // Data type for scalars
  unsigned int NDimensions = 3 >
class ITK_EXPORT GPUTranslationTransformBase : public GPUTransformBase
{
public:
  /** Standard typedefs   */
  typedef GPUTranslationTransformBase Self;
  typedef GPUTransformBase            Superclass;
  typedef SmartPointer< Self >        Pointer;
  typedef SmartPointer< const Self >  ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUTranslationTransformBase, Superclass );

  /**  */
  virtual bool IsTranslationTransform() const { return true; }

  /** Type of the scalar representing coordinate and vector elements. */
  typedef TScalarType ScalarType;

  /** Dimension of the domain space. */
  itkStaticConstMacro( SpaceDimension, unsigned int, NDimensions );
  itkStaticConstMacro( ParametersDimension, unsigned int, NDimensions );

  /** Standard vector type for this class. */
  typedef Vector< TScalarType, NDimensions > CPUOutputVectorType;

  /**  */
  virtual const CPUOutputVectorType & GetCPUOffset( void ) const = 0;

protected:
  GPUTranslationTransformBase();
  virtual ~GPUTranslationTransformBase() {}

  virtual bool GetSourceCode( std::string & _source ) const;

  virtual GPUDataManager::Pointer GetParametersDataManager() const;

private:
  GPUTranslationTransformBase( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );            // purposely not implemented

  std::vector< std::string > m_Sources;
  bool                       m_SourcesLoaded;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUTranslationTransformBase.hxx"
#endif

#endif /* itkGPUTranslationTransformBase_h */
