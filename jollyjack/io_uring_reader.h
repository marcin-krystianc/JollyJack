#pragma once

#include <arrow/io/interfaces.h>
#include <arrow/result.h>
#include <arrow/status.h>
#include <arrow/buffer.h>
#include <liburing.h>
#include <string>
#include <memory>

std::shared_ptr<arrow::io::RandomAccessFile> GetUringReader(const std::string& filename);

class IoUringReader : public arrow::io::RandomAccessFile {
 public:
  explicit IoUringReader(const std::string& filename);
  ~IoUringReader() override;

  arrow::Status Close() override;
  bool closed() const override;
  arrow::Status Seek(int64_t position) override;
  arrow::Result<int64_t> Tell() const override;
  arrow::Result<int64_t> Read(int64_t nbytes, void* out) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> Read(int64_t nbytes) override;
  arrow::Result<std::shared_ptr<arrow::Buffer>> ReadAt(int64_t position, int64_t nbytes) override;
  arrow::Result<int64_t> GetSize() override;

 private:
  int fd_;
  //static thread_local io_uring ring_;
  std::string filename_;
  int64_t pos_ = 0;
  int64_t size_ = 0;
  bool is_closed_ = false;
};