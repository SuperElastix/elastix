/*======================================================================

  This file is part of the elastix software.

  Copyright (c) University Medical Center Utrecht. All rights reserved.
  See src/CopyrightElastix.txt or http://elastix.isi.uu.nl/legal.php for
  details.

     This software is distributed WITHOUT ANY WARRANTY; without even
     the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR
     PURPOSE. See the above copyright notices for more information.

======================================================================*/
#ifndef __itkGPUCompositeTransformBase_h
#define __itkGPUCompositeTransformBase_h

#include "itkGPUTransformBase.h"
#include "itkTransform.h"

namespace itk
{
/** \class GPUCompositeTransformBaseBase
* \author Denis P. Shamonin and Marius Staring. Division of Image Processing,
* Department of Radiology, Leiden, The Netherlands
*
* This implementation was taken from elastix (http://elastix.isi.uu.nl/).
*
* \note This work was funded by the Netherlands Organisation for
* Scientific Research (NWO NRG-2010.02 and NWO 639.021.124).
*
*/
template< class TScalarType = float, unsigned int NDimensions = 3 >
class ITK_EXPORT GPUCompositeTransformBase : public GPUTransformBase
{
public:
  /** Standard class typedefs. */
  typedef GPUCompositeTransformBase  Self;
  typedef GPUTransformBase           Superclass;
  typedef SmartPointer< Self >       Pointer;
  typedef SmartPointer< const Self > ConstPointer;

  /** Run-time type information (and related methods). */
  itkTypeMacro( GPUCompositeTransformBase, Superclass );

  /** Sub transform type */
  typedef TScalarType                                        ScalarType;
  typedef Transform< TScalarType, NDimensions, NDimensions > TransformType;
  typedef typename TransformType::Pointer                    TransformTypePointer;
  typedef typename TransformType::ConstPointer               TransformTypeConstPointer;

  /** Dimension of the domain space. */
  itkStaticConstMacro( InputSpaceDimension, unsigned int, NDimensions );
  itkStaticConstMacro( OutputSpaceDimension, unsigned int, NDimensions );

  /** Get number of transforms in composite transform. */
  virtual size_t GetNumberOfTransforms() const = 0;

  /** Get the Nth transform. */
  virtual TransformTypePointer GetNthTransform( SizeValueType n ) = 0;

  /** Get the Nth transform, const version. */
  virtual TransformTypeConstPointer GetNthTransform( SizeValueType n ) const = 0;

  /**  */
  virtual bool HasIdentityTransform() const;
  virtual bool HasMatrixOffsetTransform() const;
  virtual bool HasTranslationTransform() const;
  virtual bool HasBSplineTransform() const;

  /**  */
  virtual bool IsIdentityTransform( const std::size_t index ) const;
  virtual bool IsMatrixOffsetTransform( const std::size_t index ) const;
  virtual bool IsTranslationTransform( const std::size_t index ) const;
  virtual bool IsBSplineTransform( const std::size_t index ) const;

protected:
  GPUCompositeTransformBase();
  virtual ~GPUCompositeTransformBase() {}

  virtual bool GetSourceCode( std::string & _source ) const;

  virtual GPUDataManager::Pointer GetParametersDataManager( const std::size_t index ) const;

private:
  GPUCompositeTransformBase( const Self & other ); // purposely not implemented
  const Self & operator=( const Self & );          // purposely not implemented

  bool IsIdentityTransform( const std::size_t index,
                            const bool _loadSource, std::string & _source ) const;

  bool IsMatrixOffsetTransform( const std::size_t index,
                                const bool _loadSource, std::string & _source ) const;

  bool IsTranslationTransform( const std::size_t index,
                               const bool _loadSource, std::string & _source ) const;

  bool IsBSplineTransform( const std::size_t index,
                           const bool _loadSource, std::string & _source ) const;

  std::vector< std::string > m_Sources;
  bool                       m_SourcesLoaded;
};
} // end namespace itk

#ifndef ITK_MANUAL_INSTANTIATION
#include "itkGPUCompositeTransformBase.hxx"
#endif

#endif /* __itkGPUCompositeTransformBase_h */
